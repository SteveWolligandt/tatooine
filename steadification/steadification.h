#ifndef TATOOINE_STEADIFICATION_STEADIFICATION_H
#define TATOOINE_STEADIFICATION_STEADIFICATION_H
#define TATOOINE_STEADIFICATION_PARALLEL
//#define TATOOINE_STEADIFICATION_ALL_TAUS

#include <omp.h>
#include <tatooine/chrono.h>
#include <tatooine/for_loop.h>
//#include <tatooine/gpu/reduce.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/random.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/streamsurface.h>
#include <yavin/linked_list_texture.h>

#include <boost/filesystem.hpp>
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
  using grid2_t           = grid<real_t, 2>;
  using grid3_t           = grid<real_t, 3>;
  using grid2_edge_t      = typename grid2_t::edge_t;
  using grid3_edge_t      = typename grid3_t::edge_t;
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
    GLfloat                pad;
  };
  struct rasterized_pathsurface {
    yavin::tex2rg32f            front_v;
    yavin::tex2rg32f            front_t_t0;
    yavin::tex2r32f             front_curvature;
    yavin::tex2rg32ui           front_renderindex_layer;
    yavin::tex2rg32f            back_v;
    yavin::tex2rg32f            back_t_t0;
    yavin::tex2r32f             back_curvature;
    yavin::tex2rg32ui           back_renderindex_layer;
    yavin::tex2r<std::uint32_t> list_size;
    yavin::texdepth             front_depth;
    yavin::texdepth             back_depth;
    yavin::framebuffer          front_fbo;
    yavin::framebuffer          back_fbo;
    //--------------------------------------------------------------------------
    rasterized_pathsurface(GLsizei width, GLsizei height)
        : front_v{width, height},
          front_t_t0{width, height},
          front_curvature{width, height},
          front_renderindex_layer{width, height},
          back_v{width, height},
          back_t_t0{width, height},
          back_curvature{width, height},
          back_renderindex_layer{width, height},
          list_size{width, height},
          front_depth{width, height},
          back_depth{width, height},
          front_fbo{front_v, front_t_t0, front_curvature,
                    front_renderindex_layer, front_depth},
          back_fbo{back_v, back_t_t0, back_curvature, back_renderindex_layer,
                   back_depth} {
      list_size.clear(0);
      front_fbo.bind();
      yavin::clear_color_depth_buffer();
      back_fbo.bind();
      yavin::clear_color_depth_buffer();
      back_depth.clear(1e5);
    }
    //--------------------------------------------------------------------------
   public:
    auto& front_framebuffer() { return front_fbo; }
    auto& back_framebuffer() { return back_fbo; }
  };

  //============================================================================
  // members
  //============================================================================
 private:
  const V&                  m_v;
  ivec2                     m_render_resolution;
  yavin::context            m_context;
  yavin::orthographiccamera m_cam;
  yavin::tex2rgb32f         m_color_scale;
  yavin::texdepth           m_depth;
  yavin::tex2r32f           m_noise_tex;
  yavin::tex2r32f           m_fbotex;
  yavin::framebuffer        m_fbo;
  ssf_rasterization_shader  m_ssf_rasterization_shader;
  // lic_shader                       m_lic_shader;
  // ll_to_v_shader                   m_ll_to_v_shader;
  // ll_to_curvature_shader           m_ll_to_curvature_shader;
  // weight_single_pathsurface_shader m_weight_single_pathsurface_shader;
  // weight_dual_pathsurface_shader   m_weight_dual_pathsurface_shader;
  RandEng&               m_rand_eng;
  boundingbox<real_t, 2> m_domain;
  // fragment_count_shader            m_fragment_count_shader;
  // combine_rasterizations_shader    m_combine_rasterizations_shader;
  // coverage_shader                  m_coverage_shader;
  // dual_coverage_shader             m_dual_coverage_shader;

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
        m_depth{yavin::NEAREST, yavin::CLAMP_TO_EDGE, render_resolution(0),
                render_resolution(1)},
        m_noise_tex{yavin::LINEAR, yavin::REPEAT, render_resolution(0),
                    render_resolution(1)},
        m_fbotex{render_resolution(0), render_resolution(1)},
        m_fbo{m_fbotex},
        m_rand_eng{rand_eng},
        m_domain{domain} {
    yavin::disable_multisampling();

    m_ssf_rasterization_shader.set_projection(m_cam.projection_matrix());
    // m_fragment_count_shader.set_projection(m_cam.projection_matrix());
    const auto noise_data = random_uniform_vector<float>(
        render_resolution(0) * render_resolution(1), 0.0f, 1.0f, m_rand_eng);
    m_noise_tex.upload_data(noise_data, m_render_resolution(0),
                            m_render_resolution(1));
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
  auto make_rasterization(size_t w, size_t h) {
    return rasterized_pathsurface(w, h);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <template <typename> typename SeedcurveInterpolator>
  auto rasterize(const pathsurface_t<SeedcurveInterpolator>& mesh,
                 size_t render_index, size_t layer, real_t u0t0, real_t u1t0) {
    return rasterize(gpu_pathsurface(mesh, u0t0, u1t0), render_index, layer);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& rasterize(const pathsurface_gpu_t& gpu_mesh,
                  rasterized_pathsurface& rast, size_t render_index,
                  size_t layer) {
    m_ssf_rasterization_shader.bind();
    m_ssf_rasterization_shader.set_render_index(render_index);
    m_ssf_rasterization_shader.set_layer(layer);
    yavin::gl::viewport(m_cam);
    yavin::enable_depth_test();

    rast.list_size.bind_image_texture(0);
    rast.front_framebuffer().bind();
    yavin::depth_func_less();
    m_ssf_rasterization_shader.set_count(true);
    gpu_mesh.draw();

    rast.back_framebuffer().bind();
    yavin::depth_func_greater();
    m_ssf_rasterization_shader.set_count(false);
    gpu_mesh.draw();

    return rast;
  }
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  auto pathsurface(const grid2_t& domain, size_t edge_idx, real_t u0t0,
                   real_t u1t0, real_t btau, real_t ftau, size_t seed_res,
                   real_t stepsize) const {
    const auto        edge = domain.edge_at(edge_idx);
    const seedcurve_t seedcurve{{edge.first.position(), 0},
                                {edge.second.position(), 1}};
    return pathsurface(seedcurve, u0t0, u1t0, btau, ftau, seed_res, stepsize);
  }
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  auto pathsurface(const seedcurve_t& seedcurve, real_t u0t0, real_t u1t0,
                   real_t btau, real_t ftau, size_t seed_res,
                   real_t stepsize) const {
    using namespace VC::odeint;
    integrator_t  integrator{integration::vclibs::abs_tol      = 1e-6,
                            integration::vclibs::rel_tol      = 1e-6,
                            integration::vclibs::initial_step = 0,
                            integration::vclibs::max_step     = 0.1};
    streamsurface surf{m_v, u0t0, u1t0, seedcurve, integrator};
    auto          mesh  = surf.discretize(seed_res, stepsize, btau, ftau);
    auto&         vprop = mesh.template add_vertex_property<vec2>("v");
    auto& curvprop = mesh.template add_vertex_property<real_t>("curvature");

    for (auto vertex : mesh.vertices()) {
      const auto& uv             = mesh.uv(vertex);
      const auto& integral_curve = surf.streamline_at(uv(0), 0, 0);
      curvprop[vertex]           = integral_curve.curvature(uv(1));
      if (m_v.in_domain(mesh[vertex], uv(1))) {
        vprop[vertex] =
            m_v(vec{mesh[vertex](0), mesh[vertex](1)}, mesh.uv(vertex)(1));
      } else {
        vprop[vertex] = vec<real_t, 2>{0.0 / 0.0, 0.0 / 0.0};
      }
    }
    return std::pair{std::move(mesh), std::move(surf)};
  }
  //----------------------------------------------------------------------------
  template <template <typename> typename SeedcurveInterpolator>
  auto gpu_pathsurface(
      const pathsurface_discretization_t<SeedcurveInterpolator>& mesh,
      real_t u0t0, real_t u1t0) const {
    return pathsurface_gpu_t{mesh, u0t0, u1t0};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto gpu_pathsurface(const grid2_t& domain, size_t edge_idx, real_t u0t0,
                       real_t u1t0, real_t btau, real_t ftau, size_t seed_res,
                       real_t stepsize) const {
    const auto        edge = domain.edge_at(edge_idx);
    const seedcurve_t seedcurve{{edge.first.position(), 0},
                                {edge.second.position(), 1}};
    return gpu_pathsurface(seedcurve, u0t0, u1t0, btau, ftau, seed_res,
                           stepsize);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto gpu_pathsurface(const seedcurve_t& seedcurve, real_t u0t0, real_t u1t0,
                       real_t btau, real_t ftau, size_t seed_res,
                       real_t stepsize) const {
    return gpu_pathsurface(
        pathsurface(seedcurve, u0t0, u1t0, btau, ftau, seed_res, stepsize)
            .first,
        u0t0, u1t0);
  }
  //----------------------------------------------------------------------------
  // auto weight(const rasterized_pathsurface& rast0,
  //            const rasterized_pathsurface& rast1, size_t layer1,
  //            float min_btau, float max_ftau) {
  //  yavin::tex2r32f weight_tex{m_render_resolution(0),
  //  m_render_resolution(1)}; yavin::atomiccounterbuffer
  //  num_overall_covered_pixels{0}; yavin::atomiccounterbuffer
  //  num_newly_covered_pixels{0}; rast0.bind(0, 0, 1, 0); rast1.bind(1, 2, 3,
  //  1); num_overall_covered_pixels.bind(2); num_newly_covered_pixels.bind(3);
  //  weight_tex.bind_image_texture(4);
  //  m_weight_dual_pathsurface_shader.set_linked_list0_size(rast0.buffer_size());
  //  m_weight_dual_pathsurface_shader.set_linked_list1_size(rast1.buffer_size());
  //  m_weight_dual_pathsurface_shader.set_layer(layer1);
  //  m_weight_dual_pathsurface_shader.set_min_btau(min_btau);
  //  m_weight_dual_pathsurface_shader.set_max_ftau(max_ftau);
  //  m_weight_dual_pathsurface_shader.dispatch(
  //      m_render_resolution(0) / 32.0 + 1, m_render_resolution(1) / 32.0 + 1);
  //  auto w = boost::accumulate(weight_tex.download_data(), float(0));
  //  return std::tuple{w, num_overall_covered_pixels.download_data()[0],
  //                    num_newly_covered_pixels.download_data()[0]};
  //}
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
  // auto to_lic_tex(const grid2_t& domain, float btau, float ftau,
  //                const rasterized_pathsurface& rast) {
  //  const size_t num_samples = 100;
  //  const real_t stepsize =
  //      (domain.dimension(0).back() - domain.dimension(0).front()) /
  //      (m_render_resolution(0) * 2);
  //  // generate textures
  //  auto               v_tex = to_v_tex(rast);
  //  yavin::tex2rgba32f lic_tex(m_render_resolution(0) * 2,
  //                             m_render_resolution(1) * 2);
  //
  //  lic_tex.bind_image_texture(0);
  //  v_tex.bind(0);
  //  m_noise_tex.bind(1);
  //  m_color_scale.bind(2);
  //
  //  m_lic_shader.set_domain_min(domain.front(0), domain.front(1));
  //  m_lic_shader.set_domain_max(domain.back(0), domain.back(1));
  //  m_lic_shader.set_backward_tau(btau);
  //  m_lic_shader.set_forward_tau(ftau);
  //  m_lic_shader.set_num_samples(num_samples);
  //  m_lic_shader.set_stepsize(stepsize);
  //  m_lic_shader.dispatch(m_render_resolution(0) * 2 / 32.0 + 1,
  //                        m_render_resolution(1) * 2 / 32.0 + 1);
  //
  //  return lic_tex;
  //}
  //----------------------------------------------------------------------------
  // auto to_v_tex(const rasterized_pathsurface& rast) {
  //  yavin::tex2rgba32f v_tex{m_render_resolution(0), m_render_resolution(1)};
  //  rast.bind();
  //  v_tex.bind_image_texture(2);
  //  m_ll_to_v_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
  //                            m_render_resolution(1) / 32.0 + 1);
  //  return v_tex;
  //}
  //----------------------------------------------------------------------------
  // auto to_curvature_tex(const rasterized_pathsurface& rast) {
  //  yavin::tex2rgba32f v_tex{m_render_resolution(0), m_render_resolution(1)};
  //  rast.bind();
  //  v_tex.bind_image_texture(2);
  //  m_ll_to_curvature_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
  //                                    m_render_resolution(1) / 32.0 + 1);
  //  return v_tex;
  //}
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
  // void combine(rasterized_pathsurface& rast0, rasterized_pathsurface& rast1,
  //             float min_btau, float max_ftau) {
  //  std::cerr << "  resizing rast0 buffer\n";
  //  std::cerr << "    rast0.buffer_size(): " << rast0.buffer_size() << '\n';
  //  std::cerr << "    rast1.buffer_size(): " << rast1.buffer_size() << '\n';
  //  std::cerr << "    combined size:       "
  //            << rast0.buffer_size() + rast1.buffer_size() << " ...";
  //  if (rast0.buffer_size() + rast1.buffer_size()) {
  //    std::cerr << "size_t overflow this will crash...\n";
  //  }
  //  rast0.resize_buffer(rast0.buffer_size() + rast1.buffer_size());
  //  std::cerr << " done!\n";
  //
  //  rast0.bind(0, 0, 1, 0);
  //  m_combine_rasterizations_shader.set_linked_list0_size(rast0.buffer_size());
  //
  //  rast1.bind(1, 2, 3, 1);
  //  m_combine_rasterizations_shader.set_min_btau(min_btau);
  //  m_combine_rasterizations_shader.set_max_ftau(max_ftau);
  //  m_combine_rasterizations_shader.set_linked_list1_size(rast1.buffer_size());
  //
  //  std::cerr << "  running combine shader...";
  //  m_combine_rasterizations_shader.dispatch(m_render_resolution(0) / 32.0 +
  //  1,
  //                                           m_render_resolution(1) / 32.0 +
  //                                           1);
  //  std::cerr << " done!\n";
  //}
  //----------------------------------------------------------------------------
  //  auto greedy_set_cover(const grid<real_t, 2>& domain, const real_t t0,
  //                        const real_t btau, const real_t ftau,
  //                        const size_t seed_res, const real_t stepsize,
  //                        const real_t desired_coverage) {
  //    static const float nan = 0.0f / 0.0f;
  //    std::cerr << "creating linked list textures\n";
  //    rasterized_pathsurface covered_elements{
  //        m_render_resolution(0), m_render_resolution(1), 0,
  //        linked_list_node{{nan, nan}, nan, nan, nan, 0, 0, 0xffffffff}};
  //    rasterized_pathsurface working_rast{
  //        m_render_resolution(0), m_render_resolution(1), 0,
  //        linked_list_node{{nan, nan}, nan, nan, nan, 0, 0, 0xffffffff}};
  //    yavin::atomiccounterbuffer num_overall_covered_pixels_buffer{0};
  //    yavin::atomiccounterbuffer num_newly_covered_pixels_buffer{0};
  //    yavin::tex2r32f weight_tex{m_render_resolution(0),
  //    m_render_resolution(1)}; weight_tex.bind_image_texture(4);
  //    num_overall_covered_pixels_buffer.bind(2);
  //    num_newly_covered_pixels_buffer.bind(3);
  //
  //    std::cerr << "deleting last output\n";
  //    using namespace std::filesystem;
  //    auto working_dir = std::string{settings<V>::name} + "/";
  //    if (!exists(working_dir)) { create_directory(working_dir); }
  //    for (const auto& entry : directory_iterator(working_dir)) {
  //    remove(entry); }
  //
  //    auto             best_edge_idx   = domain.num_straight_edges();
  //    auto             best_weight     = -std::numeric_limits<real_t>::max();
  //    auto             old_best_weight = best_weight;
  //    real_t           best_min_btau   = 0;
  //    real_t           best_max_ftau   = 0;
  //    size_t           render_index    = 0;
  //    size_t           layer           = 0;
  //    std::set<size_t> unused_edges;
  //    std::set<size_t> used_edges;
  //#if defined(TATOOINE_STEADIFICATION_PARALLEL)
  //    std::mutex mutex;
  //#endif
  //
  //    // set all edges as unused
  //    std::cerr << "set all edges unused\n";
  //    for (size_t edge_idx = 0; edge_idx < domain.num_straight_edges();
  //         ++edge_idx) {
  //      unused_edges.insert(edge_idx);
  //    }
  //
  //    float  cov       = 0;
  //    size_t iteration = 1;
  //    const size_t numiterations = size(unused_edges);
  //    std::cerr << "starting main loop\n";
  //    do {
  //      std::cerr << "\n== iteration " << iteration++ << " / " <<
  //      numiterations
  //                << " \n";
  //      old_best_weight = best_weight;
  //      best_weight     = -std::numeric_limits<real_t>::max();
  //      best_edge_idx   = domain.num_straight_edges();
  //
  //      std::cerr << "integrating pathsurfaces...";
  //#if defined(TATOOINE_STEADIFICATION_PARALLEL)
  //      m_context.release();
  //      auto edge_idx_it = begin(unused_edges);
  //      parallel_for_loop(
  //          [&](auto) {
  //            const auto edge_idx = *edge_idx_it++;
  //#else
  //      for (auto edge_idx : unused_edges) {
  //#endif
  //
  //        const auto mesh = pathsurface(domain, edge_idx, t0, t0, btau, ftau,
  //                                      seed_res, stepsize).first;
  //#if defined(TATOOINE_STEADIFICATION_PARALLEL)
  //        {
  //          std::lock_guard lock{mutex};
  //          m_context.make_current();
  //#endif
  //
  //              m_weight_dual_pathsurface_shader.set_linked_list0_size(
  //                  covered_elements.buffer_size());
  //              m_weight_dual_pathsurface_shader.set_layer(layer);
  //              rasterize(gpu_pathsurface(mesh, t0, t0), working_rast,
  //                        render_index, layer);
  //              covered_elements.bind(0, 0, 1, 0);
  //              working_rast.bind(1, 2, 3, 1);
  //              m_weight_dual_pathsurface_shader.set_linked_list0_size(
  //                  covered_elements.buffer_size());
  //              m_weight_dual_pathsurface_shader.set_linked_list1_size(
  //                  working_rast.buffer_size());
  ////#ifdef TATOOINE_STEADIFICATION_ALL_TAUS
  ////              for (auto cur_max_ftau : linspace{
  ////                       0.0, ftau, static_cast<size_t>(ceil(abs(ftau))) +
  ///1}) { /                for (auto cur_min_btau : linspace{ / btau, 0.0,
  ///static_cast<size_t>(ceil(abs(btau))) + 1}) {
  ////#else
  //              {
  //                {
  //                  auto cur_max_ftau = ftau;
  //                  auto cur_min_btau = btau;
  ////#endif
  //                  weight_tex.clear(0.0f);
  //                  num_overall_covered_pixels_buffer[0] = 0;
  //                  num_newly_covered_pixels_buffer[0]   = 0;
  //                  m_weight_dual_pathsurface_shader.set_min_btau(cur_min_btau);
  //                  m_weight_dual_pathsurface_shader.set_max_ftau(cur_max_ftau);
  //                  m_weight_dual_pathsurface_shader.dispatch(
  //                      m_render_resolution(0) / 32.0 + 1,
  //                      m_render_resolution(1) / 32.0 + 1);
  //
  //                  const auto weight_data = weight_tex.download_data();
  //                  auto new_weight =
  //                      std::reduce(std::execution::par, begin(weight_data),
  //                                  end(weight_data), 0.0f);
  //                  //auto new_weight = gpu::reduce(weight_tex, 16, 16);
  //
  //                  auto num_newly_covered_pixels =
  //                      num_newly_covered_pixels_buffer[0];
  //                  // auto num_overall_covered_pixels =
  //                  //    num_overall_covered_pixels_buffer[0];
  //
  //                  if (num_newly_covered_pixels > 0) {
  //                    // new_weight /= num_overall_covered_pixels;
  //
  //                    // check if mesh's seedcurve neighbors another edge
  //                    const auto unused_edge = domain.edge_at(edge_idx);
  //                    for (const auto& used_edge_idx : used_edges) {
  //                      const auto used_edge = domain.edge_at(used_edge_idx);
  //
  //                      if (used_edge.first == unused_edge.first ||
  //                          used_edge.first == unused_edge.second ||
  //                          used_edge.second == unused_edge.first ||
  //                          used_edge.second == unused_edge.second) {
  //                        new_weight *= 1.2;
  //                        break;
  //                      }
  //                    }
  //                    if (new_weight > best_weight) {
  //                      best_weight   = new_weight;
  //                      best_edge_idx = edge_idx;
  //                      best_min_btau = cur_min_btau;
  //                      best_max_ftau = cur_max_ftau;
  //                    }
  //                  }
  //                }
  //              }
  //#if defined(TATOOINE_STEADIFICATION_PARALLEL)
  //              m_context.release();
  //            }
  //          },
  //          size(unused_edges));
  //      m_context.make_current();
  //#else
  //      }
  //#endif
  //      std::cerr << "done!\n";
  //      if (best_edge_idx != domain.num_straight_edges()) {
  //        std::cerr << "combining best pathsurface...\n";
  //        combine(covered_elements,
  //                rasterize(gpu_pathsurface(domain, best_edge_idx, t0, t0,
  //                btau,
  //                                          ftau, seed_res, stepsize),
  //                          working_rast, render_index, layer),
  //                best_min_btau, best_max_ftau);
  //        std::cerr << "combining best pathsurface... done!\n";
  //        used_edges.insert(best_edge_idx);
  //
  //        std::cerr << "saving lic... ";
  //        to_lic_tex(domain, btau, ftau, covered_elements)
  //            .write_png(working_dir + "/" + std::to_string(render_index) +
  //                       ".png");
  //        std::cerr << "done!\n";
  //
  //        std::cerr << "saving current lic... ";
  //        to_lic_tex(domain, btau, ftau, covered_elements)
  //            .write_png(working_dir + "/../current.png");
  //        std::cerr << "done!\n";
  //        ++render_index;
  //      }
  //      cov = coverage(covered_elements);
  //      std::cerr << "==========\n";
  //      std::cerr << "coverage: " << cov << '\n';
  //      std::cerr << "best min btau: " << best_min_btau << '\n';
  //      std::cerr << "best max ftau: " << best_max_ftau << '\n';
  //      std::cerr << "best_weight: " << best_weight << '\n';
  //      std::cerr << "old_best_weight: " << old_best_weight << '\n';
  //      if (best_weight < old_best_weight && layer == 0) {
  //        // std::cerr << "layer = 1\n";
  //        layer = 1;
  //      }
  //      if (best_edge_idx != domain.num_straight_edges()) {
  //        unused_edges.erase(best_edge_idx);
  //      } else {
  //        // std::cerr << render_index - 1 << " bäm\n";
  //      }
  //    } while (cov < desired_coverage &&
  //             best_edge_idx != domain.num_straight_edges());
  //
  //    return covered_elements;
  //  }
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
