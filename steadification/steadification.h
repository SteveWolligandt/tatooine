#ifndef STEADIFICATION_H
#define STEADIFICATION_H

#include <tatooine/cache.h>
#include <tatooine/doublegyre.h>
#include <tatooine/grid.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/parallel_for.h>
#include <tatooine/simulated_annealing.h>
#include <tatooine/streamsurface.h>

#include <boost/filesystem.hpp>
#include <boost/range/adaptors.hpp>
#include <cstdlib>
#include <vector>

#include "gpu_mem_cache.h"
#include "linked_list_texture.h"
#include "renderers.h"
#include "shaders.h"

template <typename Real>
class Steadification {
 public:
  struct node_t {
    unsigned int next;
    float        v_x, v_y;
    float        tau;
  };

  using edge_t   = tatooine::grid_edge<Real, 2>;
  using vertex_t = tatooine::grid_vertex<Real, 2>;
  using grid_t   = tatooine::grid<Real, 2>;

  using real_vec     = std::vector<Real>;
  using edge_vec     = std::vector<edge_t>;
  using vertex_seq_t = typename grid_t::vertex_seq_t;

  //[vertex, backward_tau, forward_tau]
  using solution_t = std::vector<std::tuple<vertex_t, Real, Real>>;

  using listener_t = tatooine::simulated_annealing_listener<float, solution_t>;

  using ribbon_t     = tatooine::mesh<Real, 2>;
  using ribbon_gpu_t = StreamsurfaceRenderer;

  using ndist = std::normal_distribution<Real>;
  using udist = std::uniform_real_distribution<Real>;

  //============================================================================

 public:
  tatooine::grid<Real, 2> grid;

 private:
  tatooine::vec<size_t, 2>      render_resolution;
  const Real                    t0;
  const Real                    btau, ftau;
  size_t                        seed_res;
  Real                          stepsize;
  static constexpr unsigned int reduce_work_group_size = 1024;
  size_t                        seedcurve_length;
  size_t                        num_its;

 private:
  glfw_window                     w;
  orthographiccamera              cam;
  LinkedListTexture<node_t>       gpu_linked_list;
  tex2rgba<float>                 head_vectors;
  shaderstoragebuffer<float>      weights;
  StreamsurfaceToLinkedListShader streamsurface_to_linked_list_shader;

  tex2rgb<float>            color_scale;
  texdepth                  depth;
  tex2rgba<float>           vector_tex;
  tex2rgba<float>           lic_tex;
  tex2r<float>              noise_tex;
  tex2rgba<uint8_t>         tau_color_tex;
  tex2rgba<float>           color_lic_tex;
  texture<2, uint8_t, RGBA> combined_solutions_color_tex;

 public:
  StreamsurfaceTauShader         streamsurface_tau_shader;
  StreamsurfaceVectorfieldShader streamsurface_vectorfield_shader;
  ColorLICShader                 color_lic_shader;

 private:
  WeightShader                      weight_shader;
  SpatialCoverageShader             spatial_coverage_shader;
  LinkedListToHeadVectorsShader     linked_list_to_head_vectors_shader;
  ScreenSpaceQuad                   screen_space_quad;
  WeightShowShader                  weight_show_shader;
  LineShader                        line_shader;
  tatooine::cache<edge_t, ribbon_t> ribbon_cache;
  GPUMemCache<edge_t, ribbon_gpu_t> ribbon_gpu_cache;
  std::vector<solution_t>           solutions;

  std::mutex ribbon_mutex;
  std::mutex ribbon_gpu_mutex;

 public:
  //============================================================================
  Steadification(const tatooine::grid<Real, 2>& _grid,
                 tatooine::vec<size_t, 2> _render_resolution, Real _t0,
                 Real _btau, Real _ftau, size_t _seed_res, Real _stepsize)
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
        combined_solutions_color_tex{render_resolution(0),
                                     render_resolution(1)},

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
  float weight() {
    weight_shader.set_bw_tau(btau);
    weight_shader.set_fw_tau(ftau);
    weight_shader.dispatch2d(gpu_linked_list.w() / 16 + 1,
                             gpu_linked_list.h() / 16 + 1);

    auto ws = weights.download_data();
    return std::accumulate(begin(ws), end(ws), 0.0f);
  }
  //----------------------------------------------------------------------------
  Real angle_error(const Steadification::solution_t& sol) {
    Real error{};
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
  //------------------------------------------------------------------------------
  void show_current(const solution_t& sol) {
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
      draw_line(x, grid.dimension(1).front(), 0, x, grid.dimension(1).back(), 0,
                1, 1, 1, 0.2);

    for (auto y : grid.dimension(1))
      draw_line(grid.dimension(0).front(), y, 0, grid.dimension(0).back(), y, 0,
                1, 1, 1, 0.2);

    disable_blending();
    draw_seedline(sol, 0, 0, 0, 3);
    draw_seedline(sol, 1, 1, 1, 1);

    w.swap_buffers();
    w.poll_events();
  }
  //----------------------------------------------------------------------------
  void draw_seedline(const solution_t& sol, float r, float g, float b,
                     unsigned int width) {
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
  //---------------------------------------------------------------------------
  void draw_line(float x0, float y0, float z0, float x1, float y1, float z1,
                 float r, float g, float b, float a) {
    indexeddata<vec3> line({{x0, y0, z0}, {x1, y1, z1}}, {0, 1});
    line_shader.bind();
    line_shader.set_color(r, g, b, a);
    line.draw_lines();
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto combine_solutions_color(const vf_t& vf) {
    using namespace boost;
    using namespace adaptors;

    // set up framebuffer
    combined_solutions_color_tex.clear(1, 1, 1, 0);

    ScreenSpaceShader screen_space_shader(render_resolution[0],
                                          render_resolution[1]);
    for (const auto& sol : solutions | reversed) {
      to_lic(vf, sol, 100);
      to_tau_color_scale(vf, sol);
      to_color_lic();

      framebuffer fb{combined_solutions_color_tex};
      fb.bind();
      color_lic_tex.bind();
      enable_blending();
      blend_func_alpha();
      disable_depth_test();
      screen_space_shader.bind();
      screen_space_quad.draw();
    }
  }

  //---------------------------------------------------------------------------
  template <typename vf_t>
  void to_tau_color_scale(const vf_t& vf, const solution_t& sol) {
    framebuffer fb{tau_color_tex, depth};
    fb.bind();
    gl::clear_color(0, 0, 0, 0);
    clear_color_depth_buffer();
    disable_blending();
    enable_depth_test();

    color_scale.bind();

    streamsurface_tau_shader.bind();
    gl::viewport(0, 0, render_resolution[0], render_resolution[1]);
    for (auto it = begin(sol); it != prev(end(sol)); ++it) {
      const auto& [v0, b0, f0] = *it;
      const auto& [v1, b1, f1] = *next(it);
      streamsurface_tau_shader.set_forward_tau_range(f0, f1);
      streamsurface_tau_shader.set_backward_tau_range(b0, b1);
      ribbon_gpu(vf, {v0, v1}).draw();
    }
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto to_vectorfield_tex(const vf_t& vf, const solution_t& sol) {
    // create gpu data and program
    framebuffer fb{vector_tex, depth};
    fb.bind();
    gl::clear_color(0, 0, -1000000, 0);
    clear_color_depth_buffer();
    disable_blending();
    enable_depth_test();

    streamsurface_vectorfield_shader.bind();
    gl::viewport(0, 0, render_resolution(0), render_resolution(1));
    for (auto it = begin(sol); it != prev(end(sol)); ++it) {
      const auto& [v0, b0, f0] = *it;
      const auto& [v1, b1, f1] = *next(it);
      streamsurface_vectorfield_shader.set_forward_tau_range(f0, f1);
      streamsurface_vectorfield_shader.set_backward_tau_range(b0, b1);
      ribbon_gpu(vf, {v0, v1}).draw();
    }
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  void to_lic(const vf_t& vf, const solution_t& sol, GLuint num_steps) {
    using namespace tatooine;
    using namespace interpolation;
    const auto bb = grid.boundingbox();
    to_vectorfield_tex(vf, sol);
    lic_tex.clear(0, 0, 0, 0);

    vector_tex.bind(0);
    noise_tex.bind(1);
    lic_tex.bind_image_texture(2);

    LICShader lic_shader{num_steps, (float)bb.min(0), (float)bb.min(1),
                         (float)bb.max(0), (float)bb.max(1)};
    lic_shader.dispatch2d(render_resolution(0) / 16.0 + 1,
                          render_resolution(1) / 16.0 + 1);
  }

  //----------------------------------------------------------------------------
  template <typename layout_t>
  void seedcurve_to_tex(texture<2, float, layout_t>& tex,
                        const solution_t&            sol) {
    framebuffer fb{tex};
    fb.bind();
    draw_seedline(sol, 1, 0, 0, 5);
  }

  //----------------------------------------------------------------------------
  void to_color_lic() {
    using namespace tatooine;
    using namespace interpolation;

    const std::array res{lic_tex.width(), lic_tex.height()};
    lic_tex.bind_image_texture(2);
    tau_color_tex.bind_image_texture(3);
    color_lic_tex.bind_image_texture(4);

    color_lic_shader.dispatch2d(res[0] / 16.0 + 1, res[1] / 16.0 + 1);
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  float evaluate(const vf_t& vf, const solution_t& sol) {
    to_linked_list(vf, sol);
    show_current(sol);
    // std::this_thread::sleep_for(std::chrono::seconds{2});
    return weight();
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto ribbon_uncached(const vf_t& vf, const edge_t& e, Real stepsize) {
    using namespace VC::odeint;
    using vec2 = tatooine::vec<Real, 2>;
    tatooine::streamsurface ssf{
        vf,
        t0,
        tatooine::parameterized_line<Real, 2>{{e.first.position(), 0},
                                              {e.second.position(), 1}},
        tatooine::integration::vclibs::rungekutta43<double, 2>{
            AbsTol = 1e-6, RelTol = 1e-6, InitialStep = 0, MaxStep = stepsize},
        tatooine::interpolation::linear<Real>{},
        tatooine::interpolation::hermite<Real>{}};
    ssf.integrator().cache().set_max_memory_usage(1024 * 1024 * 25);
    auto        ribbon  = ssf.discretize(seed_res, stepsize, btau, ftau);
    const auto& mesh_uv = ribbon.template vertex_property<vec2>("uv");
    auto&       mesh_vf = ribbon.template add_vertex_property<vec2>("vf");

    for (auto v : ribbon.vertices()) {
      if (vf.in_domain(ribbon[v], t0 + mesh_uv[v](1))) {
        mesh_vf[v] = vf(ribbon[v], t0 + mesh_uv[v](1));
      } else {
        mesh_vf[v] = tatooine::vec<Real, 2>{0.0 / 0.0, 0.0 / 0.0};
      }
    }

    return ribbon;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto ribbon_uncached(const vf_t& vf, const edge_t& e) {
    return ribbon_uncached(vf, e, stepsize);
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  const auto& ribbon(const vf_t& vf, const edge_t& e, Real stepsize) {
    using namespace VC::odeint;
    if (auto found = ribbon_cache.find(e); found == ribbon_cache.end()) {
      auto            ribbon = ribbon_uncached(vf, e, stepsize);
      std::lock_guard lock(ribbon_mutex);
      return ribbon_cache.try_emplace(e, std::move(ribbon)).first->second;
    } else {
      return found->second;
    }
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  const auto& ribbon(const vf_t& vf, const edge_t& e) {
    return ribbon(vf, e, stepsize);
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto ribbons(const vf_t& vf, const solution_t& sol) {
    std::vector<ribbon_t> rs;
    for (auto it = begin(sol); it != prev(end(sol)); ++it) {
      const auto& [v0, b0, f0] = *it;
      const auto& [v1, b1, f1] = *next(it);
      rs.push_back(ribbon(vf, {v0, v1}));
    }
    return rs;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  ribbon_gpu_t ribbon_gpu_uncached(const vf_t& vf, const edge_t& e,
                                   Real stepsize) {
    return StreamsurfaceRenderer(ribbon_uncached(vf, e, stepsize));
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  ribbon_gpu_t ribbon_gpu_uncached(const vf_t& vf, const edge_t& e) {
    return ribbon_gpu_uncached(vf, e, stepsize);
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  const ribbon_gpu_t& ribbon_gpu(const vf_t& vf, const edge_t& e,
                                 Real stepsize) {
    if (auto found = ribbon_gpu_cache.find(e); found == end(ribbon_gpu_cache)) {
      return ribbon_gpu_cache
          .try_emplace(e, StreamsurfaceRenderer(ribbon(vf, e, stepsize)))
          .first->second;
    } else
      return found->second;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  const ribbon_gpu_t& ribbon_gpu(const vf_t& vf, const edge_t& e) {
    return ribbon_gpu(vf, e, stepsize);
  }

  //----------------------------------------------------------------------------
  template <size_t j>
  static auto resample_taus(const solution_t& sol, size_t n) {
    real_vec interpolated_taus(n), new_taus;
    for (size_t i = 0; i < n; ++i) {
      Real   norm_pos = Real(i) / Real(n - 1);
      auto   old_pos  = norm_pos * (sol.size() - 1);
      size_t idx      = std::floor(old_pos);
      Real   factor   = old_pos - idx;

      interpolated_taus[i] =
          std::get<j>(sol[idx]) * (1 - factor) +
          std::get<j>(sol[std::min(sol.size() - 1, idx + 1)]) * factor;
    }
    return interpolated_taus;
  };

  //------------------------------------------------------------------------
  template <typename RandEng>
  auto change_taus(real_vec taus, Real min, Real max, Real stddev,
                   RandEng&& eng) {
    //
    // {  // front
    //   auto dist = ndist{
    //       interpolated_taus.front() * 0.75 + interpolated_taus[1] * 0.25,
    //       stddev};
    //   new_taus.front() = std::max(min, std::min(max, dist(eng)));
    // }
    //
    // {  // back
    //   auto dist = ndist{
    //       interpolated_taus.back() * 0.75 +
    //           interpolated_taus[interpolated_taus.size() - 2] * 0.25,
    //       stddev};
    //   new_taus.back() = std::max(min, std::min(max, dist(eng)));
    // }
    //
    // // everything else
    // for (auto it = next(begin(interpolated_taus));
    //      it != prev(end(interpolated_taus)); ++it, ++i) {
    //   auto dist = ndist{
    //       interpolated_taus[i] * 0.5 + interpolated_taus[i - 1] * 0.25 +
    //           interpolated_taus[i + 1] * 0.25,
    //       stddev};
    //   new_taus[i] = std::max(min, std::min(max, dist(eng)));
    // }

    boost::transform(taus, begin(taus), [stddev, min, max, &eng](auto tau) {
      auto dist = ndist{tau, stddev};
      return std::max(min, std::min(max, dist(eng)));
    });

    // std::cerr << "change taus: ";
    // for (auto tau : taus) std::cerr << tau << ' ';
    // std::cerr << '\n';
    return taus;
  };

  //------------------------------------------------------------------------
  template <typename RandEng>
  static auto random_taus(size_t num, Real min, Real max, Real stddev,
                          RandEng&& eng) {
    auto     cur_tau = udist{min, max}(eng);
    real_vec taus;
    for (size_t i = 0; i < num; ++i) {
      taus.push_back(cur_tau);
      cur_tau = ndist{cur_tau, stddev / 100}(eng);
    }
    // std::cerr << "random_taus: ";
    // for (auto tau : taus) std::cerr << tau << ' ';
    // std::cerr << '\n';

    return taus;
  };

  //------------------------------------------------------------------------
  auto remove_sharp_turns(vertex_seq_t seq) {
    bool cleaned = false;
    while (!cleaned && seq.size() >= 3) {
      bool changed = false;
      for (auto it0 = begin(seq); it0 != prev(end(seq), 2) && !changed; ++it0)
        if (grid.are_direct_neighbors(*it0, *next(it0, 2))) {
          seq.erase(next(it0));
          changed = true;
        }

      if (!changed) cleaned = true;
    }

    return seq;
  };

  //------------------------------------------------------------------------
  auto clean_sequence(vertex_seq_t seq) {
    bool cleaned = false;
    while (!cleaned && seq.size() >= 4) {
      bool changed = false;
      for (auto it = begin(seq); it != prev(end(seq), 3) && !changed; ++it)
        if (grid.are_direct_neighbors(*it, *next(it, 3)) &&
            grid.are_direct_neighbors(*it, *next(it, 2)) &&
            grid.are_direct_neighbors(*next(it), *next(it, 3))) {
          seq.erase(next(it));
          changed = true;
        }

      if (!changed) cleaned = true;
    }

    cleaned = false;
    while (!cleaned && seq.size() >= 6) {
      bool changed = false;
      for (auto it0 = begin(seq); it0 != prev(end(seq), 5) && !changed; ++it0)
        for (auto it1 = next(it0, 5); it1 != end(seq) && !changed; ++it1)
          if (grid.are_direct_neighbors(*it0, *it1)) {
            seq.erase(next(it0), it1);
            changed = true;
          }

      if (!changed) cleaned = true;
    }

    return seq;
  };

  //--------------------------------------------------------------------------
  template <typename vf_t, typename RandEng>
  auto calc(const vf_t& vf, size_t _num_its, size_t _seedcurve_length,
            const std::string& path, Real desired_coverage, RandEng&& eng,
            const std::vector<listener_t*>& listeners = {}) {
    size_t         num_pixels_in_domain = 0;
    tatooine::grid render_grid{
        tatooine::linspace{grid.dimension(0).front(), grid.dimension(0).back(),
                           render_resolution(0)},
        tatooine::linspace{grid.dimension(1).front(), grid.dimension(1).back(),
                           render_resolution(1)}};
    for (auto v : render_grid.vertices()) {
      if (vf.in_domain(v.position(), t0)) { ++num_pixels_in_domain; }
    }
    std::cerr << 100.0 * Real(num_pixels_in_domain) /
                     Real(render_resolution(0) * render_resolution(1))
              << "% of pixels in domain" << '\n';
    num_its          = _num_its;
    seedcurve_length = _seedcurve_length;
    gpu_linked_list.bind();
    weights.bind(1);

    //--------------------------------------------------------------------------
    auto energy = [&vf, this](const solution_t& sol) {
      return evaluate(vf, sol);
    };

    //--------------------------------------------------------------------------
    // temperature must be between 0 and 1
    auto temperature = [this](Real i) { return 1 - i / (num_its - 1); };

    //--------------------------------------------------------------------------
    auto permute = [&, this](const solution_t& old_sol, Real temp) {
      auto stddev = temp;

      const auto global_local_border = 0.5;
      solution_t sol;
      if (temp > global_local_border) {
        // GLOBAL CHANGE
        std::uniform_real_distribution<Real> rand_cont{0, 1};
        auto                                 bw_tau = udist{btau, 0}(eng);
        auto                                 fw_tau = udist{0, ftau}(eng);
        ndist        seq_len{Real(seedcurve_length), Real(2)};
        const auto   ndist_val = abs(seq_len(eng));
        const size_t len       = std::max<size_t>(1, ndist_val);

        auto new_seq = clean_sequence(
            grid.random_straight_vertex_sequence(len, 130.0, eng));
        for (size_t i = 0; i < new_seq.size(); ++i) {
          sol.push_back(std::tuple{new_seq[i], bw_tau, fw_tau});
        }
        return sol;

      } else {
        // LOCAL CHANGE
        vertex_seq_t mutated_seq;
        boost::transform(old_sol, std::back_inserter(mutated_seq),
                         [](const auto& v) { return std::get<0>(v); });

        mutated_seq = grid.mutate_seq_straight(mutated_seq, 130, 5, eng);

        auto clamp_positive = [this](Real v) {
          return std::min(std::max<Real>(v, 0), ftau);
        };
        auto clamp_negative = [this](Real v) {
          return std::min<Real>(std::max(v, btau), 0);
        };
        auto fw_tau =
            clamp_positive(ndist{std::get<2>(old_sol.front()), stddev}(eng));
        auto bw_tau =
            clamp_negative(ndist{std::get<1>(old_sol.front()), stddev}(eng));
        for (size_t i = 0; i < mutated_seq.size(); ++i) {
          sol.push_back(std::tuple{mutated_seq[i], bw_tau, fw_tau});
        }

        return sol;
      }
    };

    auto vertex_to_tuple = [this](auto v) { return std::tuple{v, btau, ftau}; };
    size_t i             = 0;
    Real   coverage      = 0;
    while (coverage < desired_coverage) {
      solution_t start_solution;
      boost::transform(
          grid.random_straight_vertex_sequence(seedcurve_length, 130.0, eng),
          std::back_inserter(start_solution), vertex_to_tuple);

      solutions.push_back(tatooine::simulated_annealing<std::greater>(
                              start_solution, num_its, energy, temperature,
                              permute, eng, listeners)
                              .second);

      to_head_vectors(vf, solutions.back());

      gpu_linked_list.counter()[0] = 0;
      spatial_coverage_shader.dispatch2d(render_resolution(0) / 16 + 1,
                                         render_resolution(1) / 16 + 1);
      coverage = static_cast<Real>(gpu_linked_list.counter()[0]) /
                 static_cast<Real>(num_pixels_in_domain);
      std::cerr << "coverage: " << coverage * 100 << "%" << '\n';

      to_lic(vf, solutions.back(), 100);
      lic_tex.write_png(path + "result_sub" + std::to_string(i) + "_lic.png");

      to_tau_color_scale(vf, solutions.back());
      tau_color_tex.write_png(path + "result_sub" + std::to_string(i) +
                              "_color.png");

      to_color_lic();
      seedcurve_to_tex(color_lic_tex, solutions.back());
      color_lic_tex.write_png(path + "result_sub" + std::to_string(i) +
                              "_color_lic.png");

      combine_solutions_color(vf);
      combined_solutions_color_tex.write_png(path + "result.png");
      combined_solutions_color_tex.write_png("last_result_color.png");
      ++i;
    }

    return solutions;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  void to_head_vectors(const vf_t& vf, const solution_t& sol) {
    to_linked_list(vf, sol);
    linked_list_to_head_vectors_shader.dispatch2d(gpu_linked_list.w() / 16 + 1,
                                                  gpu_linked_list.h() / 16 + 1);
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  void to_linked_list(const vf_t& vf, const solution_t& sol) {
    gpu_linked_list.clear();

    vertex_seq_t seq;
    boost::transform(sol, std::back_inserter(seq),
                     [](auto& v) { return std::get<0>(v); });

    parallel_for(grid.to_edge_seq(seq), [&, this](auto e) { ribbon(vf, e); });

    streamsurface_to_linked_list_shader.bind();
    for (auto it = begin(sol); it != prev(end(sol)); ++it) {
      const auto& [v0, b0, f0] = *it;
      const auto& [v1, b1, f1] = *next(it);
      streamsurface_to_linked_list_shader.set_forward_tau_range(f0, f1);
      streamsurface_to_linked_list_shader.set_backward_tau_range(b0, b1);
      ribbon_gpu(vf, {v0, v1}).draw();
    }
  }
};

#endif
