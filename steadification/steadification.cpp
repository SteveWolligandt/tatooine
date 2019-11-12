#include <filesystem>
#include <tatooine/chrono.h>
#include <fstream>
#include "datasets.h"
#include "settings.h"
#include "random_seed.h"
#include "steadification.h"

Steadification::Steadification(const tatooine::grid<real_t, 2>& _grid,
                               tatooine::vec<size_t, 2> _render_resolution,
                               real_t _t0, real_t _btau, real_t _ftau,
                               size_t _seed_res, real_t _stepsize)
    : grid(_grid),
      render_resolution(_render_resolution),
      t0{_t0},
      btau{_btau},
      ftau{_ftau},
      seed_res{_seed_res},
      stepsize{_stepsize},
      w("steadification", render_resolution(0), render_resolution(1)),
      cam(grid.dimension(0).front(), grid.dimension(0).back(),
          grid.dimension(1).front(), grid.dimension(1).back(), btau, ftau,
          render_resolution(0), render_resolution(1)),
      gpu_linked_list(render_resolution(0), render_resolution(1)),
      head_vectors(render_resolution(0), render_resolution(1)),
      weights(render_resolution(0) * render_resolution(1)),
      streamsurface_to_linked_list_shader(gpu_linked_list.buffer_size()),

      color_scale{yavin::LINEAR, yavin::CLAMP_TO_EDGE, "color_scale.png"},
      depth{yavin::NEAREST, yavin::CLAMP_TO_EDGE, render_resolution(0),
            render_resolution(1)},
      vector_tex{render_resolution(0), render_resolution(1)},
      lic_tex{render_resolution(0), render_resolution(1)},
      noise_tex{yavin::LINEAR, yavin::REPEAT, render_resolution(0),
                render_resolution(1)},
      tau_color_tex{render_resolution(0), render_resolution(1)},
      color_lic_tex{render_resolution(0), render_resolution(1)},
      combined_solutions_color_tex{render_resolution(0), render_resolution(1)},

      weight_show_shader(render_resolution(0), render_resolution(1)),
      line_shader(1, 1, 0),
      ribbon_cache(10000, tatooine::total_memory() / 2),
      ribbon_gpu_cache(10000, tatooine::total_memory() / 2, 1 * 1024 * 1024) {
  std::cout << "render_resolution: " << render_resolution << '\n';
  yavin::disable_multisampling();
  streamsurface_to_linked_list_shader.set_projection(cam.projection_matrix());
  streamsurface_tau_shader.set_projection(cam.projection_matrix());
  streamsurface_vectorfield_shader.set_projection(cam.projection_matrix());
  line_shader.set_projection(cam.projection_matrix());
  streamsurface_tau_shader.set_tau_range({btau, ftau});

  head_vectors.clear(0, 0, 0, 0);
  gpu_linked_list.bind();
  weights.bind(1);
  head_vectors.bind_image_texture(1);

  tatooine::grid_sampler<float, 2, float, tatooine::interpolation::linear,
                         tatooine::interpolation::linear>
      noise(render_resolution(0), render_resolution(1));
  noise.randu(std::mt19937_64{1234});
  noise_tex.upload_data(noise.data().unchunk(), render_resolution(0),
                        render_resolution(1));
}

//----------------------------------------------------------------------------
real_t Steadification::angle_error(const Steadification::solution_t& sol) {
  real_t error{};
  for (auto it = begin(sol); it != prev(end(sol), 2); ++it) {
    const auto& [v0, b0, f0] = *it;
    const auto& [v1, b1, f1] = *next(it);
    const auto& [v2, b2, f2] = *next(it, 2);

    auto p0   = v0.position();
    auto p1   = v1.position();
    auto p2   = v2.position();
    auto dir0 = p0 - p1;
    auto dir1 = p2 - p1;
    error += (180.0 - acos(dot(dir0, dir1) / (norm(dir0) * norm(dir1))) *
                          180.0 / M_PI) /
             135.0;
  }
  return error / sol.size();
}

//----------------------------------------------------------------------------
float Steadification::weight() {
  weight_shader.set_bw_tau(btau);
  weight_shader.set_fw_tau(ftau);
  weight_shader.dispatch2d(gpu_linked_list.w() / 16 + 1,
                           gpu_linked_list.h() / 16 + 1);

  auto ws = weights.download_data();
  return std::accumulate(begin(ws), end(ws), 0.0f);
}

//----------------------------------------------------------------------------
void Steadification::draw_seedline(const solution_t& sol, float r, float g,
                                   float b, unsigned int width) {
  yavin::disable_depth_test();
  glLineWidth(width);
  for (auto it = begin(sol); it != prev(end(sol)); ++it) {
    const auto& [v0, b0, f0] = *it;
    const auto& [v1, b1, f1] = *next(it);
    auto first               = v0.position();
    auto second              = v1.position();
    draw_line(first(0), first(1), 0, second(0), second(1), 0, r, g, b, 1);
  }
  yavin::enable_depth_test();
}

//------------------------------------------------------------------------------
void Steadification::show_current(const solution_t& sol) {
  disable_blending();
  clear_color_depth_buffer();
  disable_depth_test();
  gl::clear_color(0, 0, 0, 0);
  gl::viewport(0, 0, render_resolution(0), render_resolution(1));
 
  weight_show_shader.bind();
  weight_show_shader.set_t0(t0);
  weight_show_shader.set_bw_tau(btau);
  weight_show_shader.set_fw_tau(ftau);
  ScreenSpaceQuad screen_space_quad;
  screen_space_quad.draw();

  enable_blending();
  blend_func_alpha();
  for (auto x : grid.dimension(0))
    draw_line(x, grid.dimension(1).front(), 0,
              x, grid.dimension(1).back(), 0,
              1,1,1, 0.2);

  for (auto y : grid.dimension(1))
    draw_line(grid.dimension(0).front(), y, 0, 
              grid.dimension(0).back(), y, 0,
              1,1,1, 0.2);

  disable_blending();
  draw_seedline(sol, 0, 0, 0, 3);
  draw_seedline(sol, 1, 1, 1, 1);

  w.swap_buffers();
  w.poll_events();
}

//---------------------------------------------------------------------------
void Steadification::draw_line(float x0, float y0, float z0, float x1, float y1,
                               float z1, float r, float g, float b, float a) {
  indexeddata<vec3> line({{x0, y0, z0}, {x1, y1, z1}}, {0, 1});
  line_shader.bind();
  line_shader.set_color(r, g, b, a);
  line.draw_lines();
}
