#ifndef __STEADIFIED_LIC_H__
#define __STEADIFIED_LIC_H__

#include <yavin>
#include <string>
#include "shaders.h"
#include <boost/range/adaptors.hpp>

struct steadified_lic {
  //----------------------------------------------------------------------------
  template <typename vf_t, typename solution_t, typename res_t>
  static auto solution_to_tau_color_scale(
      const vf_t& vf, const solution_t& sol, 
      StreamsurfaceTauShader& streamsurface_tau_shader,
      const res_t& res) {
    using namespace yavin;
    FrameBuffer              fb;
    Texture2D<uint8_t, RGBA> col(res(0), res(1));
    Texture2D<unsigned int, Depth> depth(res(0),
                                         res(1), tex::NEAREST,
                                         tex::CLAMP_TO_EDGE);
    fb.attachTex2D(col);
    fb.attachDepth(depth);
    fb.bind();
    gl::clear_color(1, 1, 1, 0);
    clear_color_depth_buffer();
    disable_blending();
    enable_depth_test();

    Texture2D<float, RGB> color_scale("color_scale.png", tex::LINEAR,
                                      tex::CLAMP_TO_EDGE);
    color_scale.bind();

    streamsurface_tau_shader.bind();
    gl::viewport(0, 0, res(0), res(1));
    for (auto it = begin(sol); it != prev(end(sol)); ++it) {
      const auto& [v0, b0, f0] = *it;
      const auto& [v1, b1, f1] = *next(it);
      streamsurface_tau_shader.set_forward_tau_range(f0, f1);
      streamsurface_tau_shader.set_backward_tau_range(b0, b1);
      ribbon_gpu(vf, {v0, v1}).draw();
    }

    return col;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t, typename solution_t, typename res_t>
  static void save_solutions(const vf_t& vf, const std::string& path,
                             const std::vector<solution_t>& solutions,
                             const res_t& res) {
    using namespace yavin;
    using namespace boost;
    using namespace adaptors;
    std::vector<Texture2D<float, RGBA>> sol_textures;
    sol_textures.reserve(solutions.size());

    transform(
        solutions, std::back_inserter(sol_textures),
        [&](const auto& sol) { return solution_to_lic(vf, sol, 100, false); });

    // set up framebuffer
    FrameBuffer              fb;
    Texture2D<uint8_t, RGBA> col(res(0), res(1));
    fb.attachTex2D(col);
    fb.bind();
    gl::clear_color(1, 1, 1, 0);
    clear_color_buffer();
    enable_blending();
    blend_func_alpha();
    disable_depth_test();

    ScreenSpaceShader screen_space_shader(res(0),
                                          res(1));
    screen_space_shader.bind();

    ScreenSpaceQuad screen_space_quad;
    for (const auto& sol_tex : sol_textures | reversed) {
      sol_tex.bind();
      screen_space_quad.draw();
    }
    col.save_png(path + "result.png");
    col.save_png("last_result.png");
    fb.unbind();
  }

  //----------------------------------------------------------------------------
  template <typename vf_t, typename solution_t, typename res_t>
  static auto solution_to_vectorfield_tex(
        const vf_t& vf, const solution_t& sol,
        StreamsurfaceVectorfieldShader& ssf_vf_shader,
        const res_t& res) {
    using namespace yavin;
    // create gpu data and program
    Texture2D<float, RGBA>         vector_tex(res(0),
                                      res(1));
    Texture2D<unsigned int, Depth> depth(res(0),
                                         res(1), tex::NEAREST,
                                         tex::CLAMP_TO_EDGE);
    FrameBuffer                    fb;
    fb.bind();
    fb.attachTex2D(vector_tex);
    fb.attachDepth(depth);
    gl::clear_color(0, 0, -1000000, 0);
    clear_color_depth_buffer();
    disable_blending();
    enable_depth_test();

    ssf_vf_shader.bind();
    gl::viewport(0, 0, res(0), res(1));
    for (auto it = begin(sol); it != prev(end(sol)); ++it) {
      const auto& [v0, b0, f0] = *it;
      const auto& [v1, b1, f1] = *next(it);
      streamsurface_vectorfield_shader.set_forward_tau_range(f0, f1);
      streamsurface_vectorfield_shader.set_backward_tau_range(b0, b1);
      ribbon_gpu(vf, {v0, v1}).draw();
    }
    return vector_tex;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t, typename solution_t, typename res_t>
  static auto solution_to_lic(const vf_t& vf, const solution_t& sol,
                              GLuint num_steps, bool with_seed,
                              StreamsurfaceTauShader& streamsurface_tau_shader,
                              StreamsurfaceVectorfieldShader& ssf_vf_shader,
                              const res_t& res) {
    using namespace yavin;
    using namespace tatooine;
    using namespace tatooine::interpolation;
    const auto                                    bb = grid.bounding();
    grid_sampler<2, float, float, linear, linear> noise(res(0),
                                                        res(1));
    noise.fill_random(std::mt19937{1234});
    Texture2D<float, RGBA> lic_tex(res(0), res(1));
    auto                   tau_color = solution_to_tau_color_scale(vf, sol, streamsurface_tau_shader, res);
    lic_tex.bind_image_texture(2);
    Texture2D<float, R> noise_tex(res(0), res(1),
                                  noise.data(), tex::LINEAR, tex::REPEAT);

    auto vf_tex = solution_to_vectorfield_tex(vf, sol, ssf_vf_shader, res);
    vf_tex.save_png("vf_tex.png");

    vf_tex.bind(0);
    noise_tex.bind(1);
    tau_color.bind(2);

    LICShader lic_shader{num_steps, (float)bb.min(0), (float)bb.min(1),
                         (float)bb.max(0), (float)bb.max(1)};
    lic_shader.dispatch2d(res(0) / 16.0,
                          res(1) / 16.0);
    if (with_seed) {
      FrameBuffer fb;
      fb.attachTex2D(lic_tex);
      fb.bind();
      draw_seedline(sol, 1, 0, 0, 5);
    }
    return lic_tex;
  }
};

#endif
