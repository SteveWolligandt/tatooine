#ifndef TATOOINE_STEADIFICATION_STEADIFICATION_H
#define TATOOINE_STEADIFICATION_STEADIFICATION_H
//#define TATOOINE_STEADIFICATION_PARALLEL

#include <cstdlib>
#include <tatooine/chrono.h>
#include <tatooine/for_loop.h>
//#include <tatooine/gpu/reduce.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/random.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/streamsurface.h>
#include <yavin/linked_list_texture.h>

#include <boost/range/adaptors.hpp>
#include <cmath>
#include <cstdlib>
#include <execution>
#include <filesystem>
#include <mutex>
#include <vector>
#include <yavin>

#include "renderers.h"
#include "shaders.h"
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename V, typename RandEng>
class steadification {
  //============================================================================
  // types
  //============================================================================
 public:
  using real_t = typename V::real_t;
  template <template <typename> typename SeedcurveInterpolator>
  using pathsurface_t =
      streamsurface<integration::vclibs::rungekutta43, SeedcurveInterpolator,
                    interpolation::hermite, V, real_t, 2>;
  template <template <typename> typename SeedcurveInterpolator =
                interpolation::linear>
  using pathsurface_discretization_t =
      hultquist_discretization<integration::vclibs::rungekutta43,
                               SeedcurveInterpolator, interpolation::hermite, V,
                               real_t, 2>;
  using pathsurface_gpu_t = streamsurface_renderer;
  using vec2              = vec<real_t, 2>;
  using vec3              = vec<real_t, 3>;
  using ivec2             = vec<size_t, 2>;
  using ivec3             = vec<size_t, 3>;
  using integrator_t =
      integration::vclibs::rungekutta43<real_t, 2, interpolation::hermite>;
  using seedcurve_t = parameterized_line<real_t, 2, interpolation::linear>;

  inline static const float nanf = 0.0f / 0.0f;
  struct node {
    yavin::vec<GLfloat, 2> v;
    GLfloat                t;
    GLfloat                t0;
    GLfloat                curvature;
    GLuint                 render_index;
    GLuint                 layer;
    GLfloat                pad;
  };
  //============================================================================
  // members
  //============================================================================
 private:
  const V&                           m_v;
  ivec2                              m_render_resolution;
  yavin::context                     m_context;
  yavin::orthographiccamera          m_cam;
  yavin::tex2rgb32f                  m_color_scale;
  yavin::tex2r32f                    m_noise_tex;
  yavin::tex2r32f                    m_fbotex;
  yavin::framebuffer                 m_fbo;
  ssf_rasterization_shader           m_ssf_rasterization_shader;
  tex_rasterization_to_buffer_shader m_tex_rasterization_to_buffer_shader;
  lic_shader                         m_lic_shader;
  ll_to_v_shader                     m_ll_to_v_shader;
  ll_to_curvature_shader             m_to_curvature_shader;
  weight_dual_pathsurface_shader     m_weight_dual_pathsurface_shader;
  RandEng&               m_rand_eng;
  boundingbox<real_t, 2> m_domain;
  combine_rasterizations_shader m_combine_rasterizations_shader;
  // coverage_shader                  m_coverage_shader;
  // dual_coverage_shader             m_dual_coverage_shader;
 public:
  yavin::tex2rgba32f  front_v_t_t0;
  yavin::tex2r32f     front_curvature;
  yavin::tex2rg32ui   front_renderindex_layer;
  yavin::texdepth32ui front_depth;
  yavin::framebuffer  front_fbo;

  yavin::tex2rgba32f  back_v_t_t0;
  yavin::tex2r32f     back_curvature;
  yavin::tex2rg32ui   back_renderindex_layer;
  yavin::texdepth32ui back_depth;
  yavin::framebuffer  back_fbo;

  yavin::tex2rgba32f lic_tex;
  yavin::tex2rgba32f color_lic_tex;
  yavin::tex2rgba32f v_tex;

  yavin::shaderstoragebuffer<node>    result_rasterization;
  yavin::shaderstoragebuffer<node>    working_rasterization;
  yavin::shaderstoragebuffer<GLuint>  result_list_size;
  yavin::shaderstoragebuffer<GLuint>  working_list_size;
  yavin::shaderstoragebuffer<GLfloat> weight_buffer;

  yavin::atomiccounterbuffer num_overall_covered_pixels{0};
  yavin::atomiccounterbuffer num_newly_covered_pixels{0};

  //============================================================================
  // ctor
  //============================================================================
 public:
  steadification(const field<V, real_t, 2, 2>& v,
                 const boundingbox<real_t, 2>& domain, ivec2 render_resolution,
                 RandEng& rand_eng)
      : m_v{v.as_derived()},
        m_render_resolution{render_resolution},
        m_context{4, 5},
        m_cam{static_cast<float>(domain.min(0)),
              static_cast<float>(domain.max(0)),
              static_cast<float>(domain.min(1)),
              static_cast<float>(domain.max(1)),
              -100000,
              100000,
              render_resolution(0),
              render_resolution(1)},
        m_color_scale{yavin::LINEAR, yavin::CLAMP_TO_EDGE, "color_scale.png"},
        m_noise_tex{yavin::LINEAR, yavin::REPEAT,
                    render_resolution(0), render_resolution(1)},
        m_fbotex{render_resolution(0), render_resolution(1)},
        m_fbo{m_fbotex},
        m_rand_eng{rand_eng},
        m_domain{domain},

        front_v_t_t0{m_render_resolution(0), m_render_resolution(1)},
        front_curvature{m_render_resolution(0), m_render_resolution(1)},
        front_renderindex_layer{m_render_resolution(0), m_render_resolution(1)},
        front_depth{m_render_resolution(0), m_render_resolution(1)},
        front_fbo{front_v_t_t0, front_curvature, front_renderindex_layer,
                  front_depth},

        back_v_t_t0{m_render_resolution(0), m_render_resolution(1)},
        back_curvature{m_render_resolution(0), m_render_resolution(1)},
        back_renderindex_layer{m_render_resolution(0), m_render_resolution(1)},
        back_depth{m_render_resolution(0), m_render_resolution(1)},
        back_fbo{back_v_t_t0, back_curvature, back_renderindex_layer,
                 back_depth},
        lic_tex{m_render_resolution(0) , m_render_resolution(1) },
        color_lic_tex{m_render_resolution(0) , m_render_resolution(1) },
        v_tex{m_render_resolution(0), m_render_resolution(1)},
        result_rasterization(
            2 * m_render_resolution(0) * m_render_resolution(1),
            {{nanf, nanf}, nanf, nanf, nanf, 0, 0, nanf}),
        working_rasterization(
            2 * m_render_resolution(0) * m_render_resolution(1),
            {{nanf, nanf}, nanf, nanf, nanf, 0, 0, nanf}),
        result_list_size(m_render_resolution(0) * m_render_resolution(1), 0),
        working_list_size(m_render_resolution(0) * m_render_resolution(1), 0),
        weight_buffer(m_render_resolution(0) * m_render_resolution(1), 0),
        num_overall_covered_pixels{0},
        num_newly_covered_pixels{0} {
    yavin::disable_multisampling();

    m_ssf_rasterization_shader.set_projection(m_cam.projection_matrix());
    m_ssf_rasterization_shader.set_width(m_render_resolution(0));
    const auto noise_data = random_uniform_vector<float>(
        render_resolution(0) * render_resolution(1), 0.0f, 1.0f, m_rand_eng);
    m_noise_tex.upload_data(noise_data, m_render_resolution(0),
                            m_render_resolution(1));

    front_v_t_t0.bind_image_texture(0);
    front_curvature.bind_image_texture(1);
    front_renderindex_layer.bind_image_texture(2);
    back_v_t_t0.bind_image_texture(3);
    back_curvature.bind_image_texture(4);
    back_renderindex_layer.bind_image_texture(5);
    lic_tex.bind_image_texture(6);
    v_tex.bind_image_texture(7);
    result_rasterization.bind(0);
    working_rasterization.bind(1);
    result_list_size.bind(2);
    working_list_size.bind(3);
    weight_buffer.bind(4);

    result_list_size.upload_data(0);
    num_overall_covered_pixels.bind(0);
    num_newly_covered_pixels.bind(1);
    v_tex.bind(0);
    m_noise_tex.bind(1);
    m_color_scale.bind(2);
    m_ssf_rasterization_shader.set_width(m_render_resolution(0));
    m_combine_rasterizations_shader.set_resolution(m_render_resolution(0),
                                                   m_render_resolution(1));
    m_lic_shader.set_v_tex_bind_point(0);
    m_lic_shader.set_noise_tex_bind_point(1);
    m_lic_shader.set_color_scale_bind_point(2);
    m_weight_dual_pathsurface_shader.set_size(m_render_resolution(0) *
                                              m_render_resolution(1));
    //working_rasterization.set_usage(yavin::DYNAMIC_COPY);
    //working_list_size.set_usage(yavin::DYNAMIC_COPY);
    yavin::gl::viewport(m_cam);
    yavin::enable_depth_test();
  }
  //============================================================================
  // methods
  //============================================================================
  template <template <typename> typename SeedcurveInterpolator>
  auto rasterize(
      const pathsurface_discretization_t<SeedcurveInterpolator>& mesh,
      size_t render_index, real_t u0t0, real_t u1t0) {
    return rasterize(gpu_pathsurface(mesh, u0t0, u1t0), render_index);
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename SeedcurveInterpolator>
  auto rasterize(const pathsurface_t<SeedcurveInterpolator>& mesh,
                 size_t render_index, size_t layer, real_t u0t0, real_t u1t0) {
    return rasterize(gpu_pathsurface(mesh, u0t0, u1t0), render_index, layer);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  void rasterize(const pathsurface_gpu_t& gpu_mesh, size_t render_index,
                  size_t layer) {
    to_tex(gpu_mesh, render_index, layer);
    to_shaderstoragebuffer();
  }
  //----------------------------------------------------------------------------
  void to_tex(const pathsurface_gpu_t& gpu_mesh, size_t render_index,
              size_t layer) {
    m_ssf_rasterization_shader.bind();
    m_ssf_rasterization_shader.set_render_index(render_index);
    m_ssf_rasterization_shader.set_layer(layer);

    yavin::shaderstorage_barrier();
    working_list_size.upload_data(0);
    yavin::shaderstorage_barrier();
    front_fbo.bind();
    yavin::clear_color_depth_buffer();
    yavin::depth_func_less();
    m_ssf_rasterization_shader.set_count(true);
    gpu_mesh.draw();

    back_fbo.bind();
    yavin::clear_color_depth_buffer();
    back_depth.clear(1e5);
    yavin::depth_func_greater();
    m_ssf_rasterization_shader.set_count(false);
    gpu_mesh.draw();
    yavin::shaderstorage_barrier();
  }
  //----------------------------------------------------------------------------
  void to_shaderstoragebuffer() {
    m_tex_rasterization_to_buffer_shader.dispatch(
        m_render_resolution(0) / 32.0 + 1, m_render_resolution(1) / 32.0 + 1);
    yavin::shaderstorage_barrier();
  }
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  auto pathsurface(const grid<real_t, 2>& domain, size_t edge_idx, real_t u0t0,
                   real_t u1t0, real_t btau, real_t ftau, size_t seed_res,
                   real_t stepsize) const {
    const auto        edge = domain.edge_at(edge_idx);
    const seedcurve_t seedcurve{{edge.first.position(), 0},
                                {edge.second.position(), 1}};
    return pathsurface(seedcurve, u0t0, u1t0, btau, ftau, seed_res, stepsize);
  }
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  auto pathsurface(const grid<real_t, 3>& domain, size_t edge_idx, real_t btau,
                   real_t ftau, size_t seed_res, real_t stepsize) const {
    const auto edge = domain.edge_at(edge_idx);
    const auto        v0   = edge.first.position();
    const auto        v1   = edge.second.position();
    const seedcurve_t seedcurve{{vec{v0(0), v0(1)}, 0}, {vec{v1(0), v1(1)}, 1}};
    return pathsurface(seedcurve, v0(2), v1(2), btau, ftau, seed_res, stepsize);
  }
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  auto pathsurface(const seedcurve_t& seedcurve, real_t u0t0, real_t u1t0,
                   real_t btau, real_t ftau, size_t seed_res,
                   real_t stepsize) const {
    using namespace VC::odeint;
    integrator_t integrator{integration::vclibs::abs_tol      = 1e-6,
                            integration::vclibs::rel_tol      = 1e-6,
                            integration::vclibs::initial_step = 0,
                            integration::vclibs::max_step     = 0.1};
    streamsurface surf{m_v, u0t0, u1t0, seedcurve, integrator};
    if (u0t0 == u1t0) {
    simple_tri_mesh<real_t, 2> mesh =
        surf.discretize(seed_res, stepsize, btau, ftau);
    auto& uvprop   = mesh.template add_vertex_property<vec2>("uv");
    auto& vprop    = mesh.template add_vertex_property<vec2>("v");
    auto& curvprop = mesh.template add_vertex_property<real_t>("curvature");

    for (auto vertex : mesh.vertices()) {
      const auto& uv             = uvprop[vertex];
      const auto& integral_curve = surf.streamline_at(uv(0), 0, 0);
      curvprop[vertex]           = integral_curve.curvature(uv(1));
      if (m_v.in_domain(mesh[vertex], uv(1))) {
        vprop[vertex] =
            m_v(vec{mesh[vertex](0), mesh[vertex](1)}, uvprop[vertex](1));
      } else {
        vprop[vertex] = vec<real_t, 2>{0.0 / 0.0, 0.0 / 0.0};
      }
    }
    return std::pair{std::move(mesh), std::move(surf)};
    } else {
      return std::pair{simple_tri_mesh<real_t, 2>{}, std::move(surf)};
    }
  }
  //----------------------------------------------------------------------------
  auto gpu_pathsurface(simple_tri_mesh<real_t, 2>& mesh, real_t u0t0,
                       real_t u1t0) const {
    return pathsurface_gpu_t{mesh, u0t0, u1t0};
  }
  //----------------------------------------------------------------------------
  auto weight(GLuint layer1) {
    weight_buffer.upload_data(0.0f);
    num_overall_covered_pixels[0] = 0;
    num_newly_covered_pixels[0]   = 0;
    m_weight_dual_pathsurface_shader.set_layer(layer1);
    m_weight_dual_pathsurface_shader.bind();
    yavin::gl::dispatch_compute(
        m_render_resolution(0) * m_render_resolution(1) / 1024.0 + 1, 1, 1);

    const auto weight_data = weight_buffer.download_data();
    return std::reduce(std::execution::par, begin(weight_data),
                       end(weight_data), 0.0f);
    // return gpu::reduce(weight_buffer, 16, 16);
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename SeedcurveInterpolator>
  auto curvature(
      const pathsurface_discretization_t<SeedcurveInterpolator>& mesh,
      const pathsurface_t<SeedcurveInterpolator>&                surf) const {
    std::set<real_t> us;
    for (auto v : mesh.vertices()) { us.insert(mesh.uv(v)(0)); }
    const auto num_integral_curves = us.size();

    std::vector<real_t> kappas, arc_lengths;
    arc_lengths.reserve(num_integral_curves);
    kappas.reserve(num_integral_curves);
    size_t i = 0;
    for (auto u : us) {
      const auto& integral_curve = surf.streamline_at(u, 0, 0);
      kappas.push_back(integral_curve.integrate_curvature());
      arc_lengths.push_back(integral_curve.arc_length());
      ++i;
    }
    real_t acc_kappas = 0;
    for (size_t i = 0; i < num_integral_curves; ++i) {
      acc_kappas += kappas[i] * arc_lengths[i];
    }
    return acc_kappas / boost::accumulate(arc_lengths, real_t(0));
  }
  //----------------------------------------------------------------------------
  template <size_t N>
  void result_to_lic_tex(const grid<real_t, N>& domain, GLfloat btau, GLfloat ftau) {
    const size_t num_samples = 20;
    const real_t stepsize =
        (domain.dimension(0).back() - domain.dimension(0).front()) /
        (m_render_resolution(0) * 2);
    result_to_v_tex();

    color_lic_tex.bind_image_texture(7);
    lic_tex.clear(0, 0, 0, 0);
    color_lic_tex.clear(0, 0, 0, 0);
    m_lic_shader.set_domain_min(domain.front(0), domain.front(1));
    m_lic_shader.set_domain_max(domain.back(0), domain.back(1));
    m_lic_shader.set_backward_tau(btau);
    m_lic_shader.set_forward_tau(ftau);
    m_lic_shader.set_num_samples(num_samples);
    m_lic_shader.set_stepsize(stepsize);
    m_lic_shader.dispatch(m_render_resolution(0)  / 32.0 + 1,
                          m_render_resolution(1)  / 32.0 + 1);
    v_tex.bind_image_texture(7);
  }
  //----------------------------------------------------------------------------
  auto result_to_v_tex() {
    m_ll_to_v_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                              m_render_resolution(1) / 32.0 + 1);
  }
  //----------------------------------------------------------------------------
  auto result_rasterization_to_curvature_tex() {
    yavin::tex2r32f curvature_tex{m_render_resolution(0),
                                  m_render_resolution(1)};
    curvature_tex.bind_image_texture(7);
    m_to_curvature_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                   m_render_resolution(1) / 32.0 + 1);
    v_tex.bind_image_texture(7);
    return curvature_tex;
  }
  //----------------------------------------------------------------------------
  // auto coverage(rasterized_pathsurface& rast) {
  //  return static_cast<real_t>(num_covered_pixels(rast)) /
  //         (m_render_resolution(0) * m_render_resolution(1));
  //}
  //----------------------------------------------------------------------------
  // auto coverage(rasterized_pathsurface& rast0, rasterized_pathsurface& rast1)
  // {
  //  return static_cast<real_t>(num_covered_pixels(rast0, rast1)) /
  //         (m_render_resolution(0) * m_render_resolution(1));
  //}
  //----------------------------------------------------------------------------
  // auto num_covered_pixels(rasterized_pathsurface& rast) {
  //  yavin::atomiccounterbuffer cnt{0};
  //  cnt.bind(1);
  //  rast.bind(0, 0, 1, 0);
  //
  //  m_coverage_shader.set_linked_list_size(rast.buffer_size());
  //  m_coverage_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
  //                             m_render_resolution(1) / 32.0 + 1);
  //  return cnt.download_data()[0];
  //}
  //----------------------------------------------------------------------------
  // auto num_covered_pixels(rasterized_pathsurface& rast0,
  //                        rasterized_pathsurface& rast1) {
  //  yavin::atomiccounterbuffer cnt{0};
  //  cnt.bind(2);
  //  rast0.bind(0, 0, 1, 0);
  //  m_dual_coverage_shader.set_linked_list0_size(rast0.buffer_size());
  //
  //  rast1.bind(1, 2, 3, 1);
  //  m_dual_coverage_shader.set_linked_list1_size(rast1.buffer_size());
  //
  //  m_dual_coverage_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
  //                                  m_render_resolution(1) / 32.0 + 1);
  //  return cnt.download_data()[0];
  //}
  //----------------------------------------------------------------------------
  /// rast1 gets written in rast0. rast0 must have additional space to be able
  /// to hold rast1.
  void combine() {
    m_combine_rasterizations_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                             m_render_resolution(1) / 32.0 + 1);
    yavin::shaderstorage_barrier();
  }
  //----------------------------------------------------------------------------
  template <size_t N>
  auto integrate(const std::string&      dataset_name,
                 const std::set<size_t>& unused_edges,
                 const grid<real_t, N>& domain, const real_t t0,
                 const real_t btau, const real_t ftau, const size_t seed_res,
                 const real_t stepsize) {
    if (!std::filesystem::exists("pathsurfaces")) {
      std::filesystem::create_directory("pathsurfaces");
    }

    const auto pathsurface_dir = +"pathsurfaces/" + dataset_name + "/";
    if (!std::filesystem::exists(pathsurface_dir)) {
      std::filesystem::create_directory(pathsurface_dir);
    }

    std::string filename_vtk;
    std::atomic_size_t progress_counter = 0;
    std::thread        t{[&] {
      float progress = 0.0;
      const int bar_width = 10;
      std::cerr << "integrating pathsurfaces...\n";
      while (progress < 1.0) {
        progress = float(progress_counter) / (unused_edges.size());

        int pos = bar_width * progress;
        for (int i = 0; i < bar_width; ++i) {
          if (i < pos)
            std::cerr << "\u2588";
          else if (i == pos)
            std::cerr << "\u2592";
          else
            std::cerr << "\u2591";
        }
        std::cerr << " " << int(progress * 100.0) << " % - " << filename_vtk
                  << "        \r";
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
      }
        for (int i = 0; i < bar_width; ++i) { std::cerr << "\u2588"; }
        std::cerr << "done!                  \n";
    }};

#if defined(TATOOINE_STEADIFICATION_PARALLEL)
    auto edge_idx_it = begin(unused_edges);
      parallel_for_loop([&](auto) {
      const auto edge_idx = *edge_idx_it++;
#else
    for (auto edge_idx : unused_edges) {
#endif

      filename_vtk = pathsurface_dir;
      for (size_t i = 0; i < N; ++i) {
        filename_vtk += std::to_string(domain.size(i)) + "_";
      }
      filename_vtk += std::to_string(t0) + "_" + std::to_string(btau) + "_" +
                      std::to_string(ftau) + "_" + std::to_string(seed_res) +
                      "_" + std::to_string(stepsize) + "_" +
                      std::to_string(edge_idx) + ".vtk";
      if (!std::filesystem::exists(filename_vtk)) {
        if constexpr (N == 2) {
          simple_tri_mesh<real_t, 2> psf =
              pathsurface(domain, edge_idx, t0, t0, btau, ftau, seed_res,
                          stepsize)
                  .first;
          psf.write_vtk(filename_vtk);
        } else if constexpr (N == 3) {
          simple_tri_mesh<real_t, 2> psf =
              pathsurface(domain, edge_idx, btau, ftau, seed_res, stepsize)
                  .first;
          psf.write_vtk(filename_vtk);
        }
      }
      progress_counter++;
#if !defined(TATOOINE_STEADIFICATION_PARALLEL)
      }
#else
    }, size(unused_edges));
#endif
      t.join();
    return pathsurface_dir;
  }
  //----------------------------------------------------------------------------
  auto setup_working_dir(const real_t t0, const real_t btau, const real_t ftau,
                         const size_t seed_res, const real_t stepsize,
                         const double neighbor_weight, const float penalty) {
    using namespace std::filesystem;
    const auto working_dir =
        std::string{settings<V>::name} + "_t0_" + std::to_string(t0) +
        "_btau_" + std::to_string(btau) + "_ftau_" + std::to_string(ftau) +
        "_seedres_" + std::to_string(seed_res) + "_stepsize_" +
        std::to_string(stepsize) + "_neighborweight_" +
        std::to_string(neighbor_weight) + "_penalty_" +
        std::to_string(penalty) + "/";
    if (!exists(working_dir)) { create_directory(working_dir); }
    std::cerr << "result will be located in " << working_dir << '\n';

    for (const auto& entry : directory_iterator(working_dir)) { remove(entry); }
    return working_dir;
  }
  //----------------------------------------------------------------------------
  template <size_t N>
  auto greedy_set_cover(const grid<real_t, N>& domain, const real_t t0,
                        const real_t btau, const real_t ftau,
                        const size_t seed_res, const real_t stepsize,
                        const real_t desired_coverage,
                        const double neighbor_weight, const float penalty) {
    using namespace std::filesystem;
    auto             best_edge_idx   = domain.num_straight_edges();
    auto             best_weight     = -std::numeric_limits<real_t>::max();
    auto             old_best_weight = best_weight;
    size_t           render_index    = 0;
    size_t           layer           = 0;
    std::set<size_t> unused_edges;
    std::vector<size_t> used_edges;
    size_t       iteration     = 0;
    size_t       edge_counter  = 0;
    bool         stop_thread   = false;
    double       coverage      = 0;
    std::vector<float> weights;
    std::vector<float> coverages;

    auto working_dir =
        setup_working_dir(t0, btau, ftau, seed_res, stepsize,
                          neighbor_weight, penalty);
    m_weight_dual_pathsurface_shader.set_penalty(penalty);

    // set all edges as unused
    std::cerr << "set all edges unused\n";
    for (size_t edge_idx = 0; edge_idx < domain.num_straight_edges();
         ++edge_idx) {
      unused_edges.insert(edge_idx);
    }

    const auto pathsurface_dir =
        integrate(std::string{settings<V>::name}, unused_edges, domain, t0,
                  btau, ftau, seed_res, stepsize);

    // monitoring
    std::thread t{[&] {
      double prog0 = 0.0;
      double prog1 = 0.0;
      std::cerr << "cur it          used edges      coverage    \n";
      while (!stop_thread) {
        const int bar_width = 15;
        prog0              = double(edge_counter) / (unused_edges.size());
        prog1              = double(iteration) / (domain.num_straight_edges());
        int pos0 = bar_width * prog0;
        int pos1 = bar_width * prog1;
        int pos2 = bar_width * coverage;
        for (int i = 0; i < bar_width; ++i) {
          if (i < pos0)
            std::cerr << "\u2588";
          else if (i == pos0)
            std::cerr << "\u2592";
          else
            std::cerr << "\u2591";
        }
        std::cerr << " ";
        for (int i = 0; i < bar_width; ++i) {
          if (i < pos1)
            std::cerr << "\u2588";
          else if (i == pos1)
            std::cerr << "\u2592";
          else
            std::cerr << "\u2591";
        }
        std::cerr << " ";
        for (int i = 0; i < bar_width; ++i) {
          if (i < pos2)
            std::cerr << "\u2588";
          else if (i == pos2)
            std::cerr << "\u2592";
          else
            std::cerr << "\u2591";
        }
        std::cerr << "\r";
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
      }
      std::cerr << "\n";
    }};

    //std::vector<node>   working_rasterization_data(m_render_resolution(0) *
    //                                             m_render_resolution(1));
    //std::vector<GLuint> working_list_size_data(m_render_resolution(0) *
    //                                           m_render_resolution(1));

    // loop
    do {
      edge_counter    = 0;
      old_best_weight = best_weight;
      best_weight     = -std::numeric_limits<real_t>::max();
      best_edge_idx   = domain.num_straight_edges();

      for (auto edge_idx : unused_edges) {
        std::string filepath_vtk = pathsurface_dir;
        for (size_t i = 0; i < N; ++i) {
          filepath_vtk += std::to_string(domain.size(i)) + "_";
        }
        filepath_vtk += std::to_string(t0) + "_" + std::to_string(btau) + "_" +
                        std::to_string(ftau) + "_" + std::to_string(seed_res) +
                        "_" + std::to_string(stepsize) + "_" +
                        std::to_string(edge_idx) + ".vtk";
        simple_tri_mesh<real_t, 2> mesh{filepath_vtk};
        if (mesh.num_faces() > 0) {
          rasterize(gpu_pathsurface(mesh, t0, t0), render_index, layer);
          // std::ifstream file{filepath_buf};
          // if (file.is_open()) {
          //  file.read((char*)working_rasterization_data.data(),
          //            working_rasterization_data.size() *
          //            sizeof(node));
          //  file.read((char*)working_list_size_data.data(),
          //            working_list_size_data.size() *
          //            sizeof(GLuint));
          //  file.close();
          //}
          // working_rasterization.upload_data(working_rasterization_data);
          // working_list_size.upload_data(working_list_size_data);
          auto new_weight = weight(layer);
          if (num_newly_covered_pixels[0] > 0) {
            // new_weight /= num_overall_covered_pixels;

            // check if mesh's seedcurve neighbors another edge
            const auto unused_edge = domain.edge_at(edge_idx);
            size_t     num_usages0 = 0, num_usages1 = 0;
            for (const auto used_edge_idx : used_edges) {
              const auto [uv0, uv1] = domain.edge_at(used_edge_idx);

              if (uv0 == unused_edge.first || uv1 == unused_edge.first) {
                ++num_usages0;
              }
              if (uv1 == unused_edge.second || uv1 == unused_edge.second) {
                ++num_usages1;
              }
            }
            if (num_usages0 > 1 || num_usages1 > 1) {
              new_weight /= neighbor_weight;
            } else if (num_usages0 == 1 || num_usages1 == 1) {
              new_weight *= neighbor_weight;
            }
            if (new_weight > best_weight) {
              best_weight   = new_weight;
              best_edge_idx = edge_idx;
            }
          }
        }
        ++edge_counter;
      }
      if (best_edge_idx != domain.num_straight_edges()) {
        std::string filepath = pathsurface_dir;
        for (size_t i = 0; i < N; ++i) {
          filepath += std::to_string(domain.size(i)) + "_";
        }
        filepath += std::to_string(t0) + "_" + std::to_string(btau) + "_" +
                    std::to_string(ftau) + "_" + std::to_string(seed_res) +
                    "_" + std::to_string(stepsize) + "_" +
                    std::to_string(best_edge_idx) + ".vtk";
        simple_tri_mesh<real_t, 2> mesh{filepath};
        rasterize(gpu_pathsurface(mesh, t0, t0), render_index, layer);
        combine();
        used_edges.push_back(best_edge_idx);

        ++render_index;
        coverage =
            static_cast<double>(num_overall_covered_pixels[0].download()) /
            (m_render_resolution(0) * m_render_resolution(1));
        weights.push_back(best_weight /
                          (m_render_resolution(0) * m_render_resolution(1)));
        coverages.push_back(coverage);

        std::string it_str = std::to_string(iteration);
        while (it_str.size() < 4) { it_str = '0' + it_str; }
        result_to_lic_tex(domain, btau, ftau);
        lic_tex.write_png(working_dir + "lic_" + it_str + ".png");
        color_lic_tex.write_png(working_dir + "lic_color_" + it_str + ".png");
        std::string mesh2dpath = pathsurface_dir;
        for (size_t i = 0; i < N; ++i) {
          mesh2dpath += std::to_string(domain.size(i)) + "_";
        }
        mesh2dpath += std::to_string(t0) + "_" + std::to_string(btau) + "_" +
                      std::to_string(ftau) + "_" + std::to_string(seed_res) +
                      "_" + std::to_string(stepsize) + "_" +
                      std::to_string(best_edge_idx) + ".vtk";
        const std::string mesh3dpath =
            working_dir + "geometry_" + it_str + ".vtk";
        simple_tri_mesh<real_t, 2> mesh2d{mesh2dpath};
        simple_tri_mesh<real_t, 3> mesh3d;
        auto& uv2d_prop = mesh2d.template vertex_property<vec<real_t, 2>>("uv");
        auto& uv3d_prop =
            mesh3d.template add_vertex_property<vec<real_t, 2>>("uv");
        auto& curv2d_prop =
            mesh2d.template vertex_property<real_t>("curvature");
        auto& curv3d_prop =
            mesh3d.template add_vertex_property<real_t>("curvature");
        auto& v2d_prop = mesh2d.template vertex_property<vec<real_t, 2>>("v");
        auto& v3d_prop =
            mesh3d.template add_vertex_property<vec<real_t, 2>>("v");

        for (const auto v : mesh2d.vertices()) {
          const auto& x = mesh2d[v];
          mesh3d.insert_vertex(x(0), x(1), uv2d_prop[v](1));
          uv3d_prop[v.i]   = uv2d_prop[v];
          curv3d_prop[v.i] = curv2d_prop[v];
          v3d_prop[v.i]    = v2d_prop[v];
        }
        for (const auto f : mesh2d.faces()) {
          const auto& [v0, v1, v2] = mesh2d[f];
          mesh3d.insert_face(v0.i, v1.i, v2.i);
        }

        mesh3d.write_vtk(mesh3dpath);
      }
      // if (best_weight < old_best_weight && layer == 0) { layer =
      // 1; }
      if (best_edge_idx != domain.num_straight_edges()) {
        unused_edges.erase(best_edge_idx);
      }
      ++iteration;
    }
    while (coverage < desired_coverage &&
           best_edge_idx != domain.num_straight_edges())
      ;
    result_to_lic_tex(domain, btau, ftau);
    lic_tex.write_png(working_dir + "lic_final.png");
    color_lic_tex.write_png(working_dir + "lic_color_final.png");
    std::vector<line<real_t, 3>> lines;
    for (auto used_edge:used_edges) {
      auto e = domain.edge_at(used_edge);
      auto v0 = e.first.position();
      auto v1 = e.second.position();
      if constexpr (N == 2) {
        lines.push_back(line<real_t, 3>{vec<real_t, 3>{v0(0), v0(1), t0},
                                        vec<real_t, 3>{v1(0), v1(1), t0}});
      } else if constexpr (N == 3) {
        lines.push_back(line<real_t, 3>{v0, v1});
      }
    }
    write_vtk(lines, working_dir + "seeds.vtk");
    stop_thread = true;
    t.join();
    std::cerr << '\n';

    std::string cmd = "#/bin/bash \n";
    cmd += "cd " + working_dir + '\n';
    cmd +=
        "ffmpeg -y -r 3 -start_number 0 -i lic_%04d.png -c:v libx264 -vf fps=25 "
        "-pix_fmt yuv420p lic.mp4\n";
    cmd +=
        "ffmpeg -y -r 3 -start_number 0 -i lic_color_%04d.png -c:v libx264 -vf "
        "fps=25 "
        "-pix_fmt yuv420p lic_color.mp4\n";
    system(cmd.c_str());

    const std::string reportfilepath = working_dir + "report.html";
    std::ofstream     reportfile{reportfilepath};
    reportfile
        << "<!DOCTYPE html>\n"
        << "<html><head>\n"
        << "<meta charset=\"utf-8\">\n"
        << "<meta name=\"viewport\" content=\"width=device-width, "
           "initial-scale=1, shrink-to-fit=no\">\n"
        << "<link rel=\"stylesheet\" "
           "href=\"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/"
           "css/bootstrap.min.css\" "
           "integrity=\"sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/"
           "iJTQUOhcWr7x9JvoRxT2MZw1T\" crossorigin=\"anonymous\">\n"
        << "<script "
           "src=\"https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.5.0/"
           "Chart.min.js\"></script>\n"
        << "<style type=\"text/css\" title=\"text/css\">\n"
        << "#outerwrap {\n"
        << "  margin:auto;\n"
        << "  width:800px;\n"
        << "  border:1px solid #CCCCCC;\n"
        << "  padding 10px;\n"
        << "}\n"
        << "#innerwrap {\n"
        << "  margin:10px;\n"
        << "}\n"
        << "</style>\n"

        << "</head><body>\n"
        << "<script src=\"https://code.jquery.com/jquery-3.3.1.slim.min.js\" "
           "integrity=\"sha384-q8i/"
           "X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo\" "
           "crossorigin=\"anonymous\"></script>\n"
        << "<script "
           "src=\"https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/"
           "popper.min.js\" "
           "integrity=\"sha384-"
           "UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1\" "
           "crossorigin=\"anonymous\"></script>\n"
        << "<script "
           "src=\"https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/"
           "bootstrap.min.js\" "
           "integrity=\"sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/"
           "nJGzIxFDsf4x0xIM+B07jRM\" crossorigin=\"anonymous\"></script>\n"
        << "<script "
           "src=\"https://cdn.jsdelivr.net/npm/chart.js@2.8.0\"></script>\n"
        << "<div id=\"outerwrap\"><div id=\"innerwrap\">\n"
        << "\n"
        << "<table class=\"table\">\n"
        << "<tr><th>t0</th><th>backward tau</th><th>forward tau</th><th>seed "
           "res</th><th>stepsize</th><th>coverage</th><th>neighbor "
           "weight</th><th>penalty</th><th>num_iterations</th></tr>\n"
        << "<tr><td>" << t0 << "</td><td>" << btau << "</td><td>" << ftau
        << "</td><td>" << seed_res << "</td><td>" << stepsize << "</td><td>"
        << desired_coverage << "</td><td>" << neighbor_weight << "</td><td>"
        << penalty << "</td><td>" << used_edges.size() << "</td></tr>\n"

        << "</table>\n"
        << "\n"
        << "<table><tr>\n"
        << "<td><img width=100% src=\"lic_final.png\"></img></td>\n"
        << "<td><img width=100% src=\"lic_color_final.png\"></img></td>\n"
        << "</tr></table>\n"
        << "\n"
        << "<table><tr>\n"
        << "<td><video width=\"100%\" controls><source src=\"lic.mp4\" "
           "type=\"video/mp4\"></video></td>\n"
        << "<td><video width=\"100%\" controls><source src=\"lic_color.mp4\" "
           "type=\"video/mp4\"></video></td>\n"
        << "</tr></table>\n"
        << '\n'
        << "<div id=\"carouselExampleIndicators\" class=\"carousel\">\n"
        << "  <ol class=\"carousel-indicators\">\n"
        << "    <li data-target=\"#carouselExampleIndicators\" "
           "data-slide-to=\"0\" class=\"active\"></li>\n";

    for (size_t i = 1; i < used_edges.size(); ++i) {
      reportfile << "<li data-target=\"#carouselExampleIndicators\" "
                    "data-slide-to=\""
                 << i << "\"></li>\n";
    }

    reportfile 
      << "</ol>\n"
      << "<div class=\"carousel-inner\">\n"
      << "<div class=\"carousel-item active\">\n"
      << "<table class=\"table\">\n"
      << "<tr><th>iteration#</th><th>weight</th><th>coverage</th></tr>\n"
      << "<tr><td>"<<0<<"</td><td>"<<weights[0]<<"</td><td>"<<coverages[0]<<"</td></tr>\n"
      << "</table>\n"
      << "  <img width=100% src=\"lic_0000.png\">\n"
      << "  <img width=100% src=\"lic_color_0000.png\">\n"
      << "</div>\n";
      for (size_t i = 1; i < used_edges.size(); ++i) {
        std::string itstr = std::to_string(i);
        while (itstr.size() < 4) {itstr = '0' + itstr;}
        reportfile
          << "<div class=\"carousel-item\">\n"
          << "<table class=\"table\">\n"
          << "<tr><th>iteration#</th><th>weight</th><th>coverage</th></tr>\n"
          << "<tr><td>"<<i<<"</td><td>"<<weights[i]<<"</td><td>"<<coverages[i]<<"</td></tr>\n"
          << "</table>\n"
          << "  <img width=100% src=\"lic_"<<itstr<<".png\">\n"
          << "  <img width=100% src=\"lic_color_"<<itstr<<".png\">\n"
          << "</div>\n";
      }

      reportfile << "</div>\n"
                 << "<a class=\"carousel-control-prev\" "
                    "href=\"#carouselExampleIndicators\" role=\"button\" "
                    "data-slide=\"prev\">\n"
                 << "<span class=\"carousel-control-prev-icon\" "
                    "aria-hidden=\"true\"></span>\n"
                 << "<span class=\"sr-only\">Previous</span>\n"
                 << "</a>\n"
                 << "<a class=\"carousel-control-next\" "
                    "href=\"#carouselExampleIndicators\" role=\"button\" "
                    "data-slide=\"next\">\n"
                 << "<span class=\"carousel-control-next-icon\" "
                    "aria-hidden=\"true\"></span>\n"
                 << "<span class=\"sr-only\">Next</span>\n"
                 << "</a>\n"
                 << "</div>\n";

      reportfile
          << "<canvas id=\"weight-chart\" width=100%></canvas>\n"
          << "<script>\n"
          << "new Chart(document.getElementById(\"weight-chart\"), {\n"
          << "type: 'line',\n"
          << "data: {\n"
          << "labels: [" << 0;
      for (size_t i = 1; i < used_edges.size(); ++i) {
        reportfile << ", " << i;
      }
      reportfile << "],\n"
          << "datasets: [{ \n"
          << "data: [" << weights.front();

      for (size_t i = 1; i < used_edges.size(); ++i) {
        reportfile << ", " << weights[i];
      }
      reportfile << "],\n"
                 << "label: \"weights\",\n"
                 << "borderColor: \"#3e95cd\",\n"
                 << "fill: false\n"
                 << "}\n"
                 << "]\n"
                 << "},\n"
                 << "options: {\n"
                 << "title: {\n"
                 << "display: true,\n"
                 << "text: 'weights'\n"
                 << "}\n"
                 << "}\n"
                 << "});</script>\n";

      reportfile
          << "<canvas id=\"coverage-chart\" width=100%></canvas>\n"
          << "<script>\n"
          << "new Chart(document.getElementById(\"coverage-chart\"), {\n"
          << "type: 'line',\n"
          << "data: {\n"
          << "labels: [" << 0;
      for (size_t i = 1; i < used_edges.size(); ++i) {
        reportfile << ", " << i;
      }
      reportfile << "],\n"
          << "datasets: [{ \n"
          << "data: [" << coverages.front();
      for (size_t i = 1; i < used_edges.size(); ++i) {
        reportfile << ", " << coverages[i];
      }
      reportfile << "],\n"
                 << "label: \"coverage\",\n"
                 << "borderColor: \"#FF0000\",\n"
                 << "fill: false\n"
                 << "}\n"
                 << "]\n"
                 << "},\n"
                 << "options: {\n"
                 << "title: {\n"
                 << "display: true,\n"
                 << "text: 'coverage'\n"
                 << "}\n"
                 << "}\n"
                 << "});</script>\n"
                 << "</div></div></body></html>\n";
      return result_rasterization;
  }
};
  //==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
