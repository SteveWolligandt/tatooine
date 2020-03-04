#ifndef STEADIFICATION_STEADIFICATION_H
#define STEADIFICATION_STEADIFICATION_H

#include <tatooine/for_loop.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/random.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/streamsurface.h>

#include <boost/filesystem.hpp>
#include <boost/range/adaptors.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include <yavin>

#include "linked_list_texture.h"
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
      integration::vclibs::rungekutta43<double, 2, interpolation::hermite>;
  using seedcurve_t = parameterized_line<double, 2, interpolation::linear>;
  using domain_coverage_tex_t = yavin::tex2r32ui;

  struct linked_list_node {
    vec<float, 2> pos;
    vec<float, 2> v;
    vec<float, 2> uv;
    float         curvature;
    unsigned int  next;
  };
  using rasterized_pathsurface = linked_list_texture<linked_list_node>;

  //============================================================================
  // members
  //============================================================================
 private:
  const V&                         m_v;
  ivec2                            m_render_resolution;
  yavin::context                   m_context;
  yavin::orthographiccamera        m_cam;
  yavin::tex2rgb32f                m_color_scale;
  yavin::texdepth                  m_depth;
  yavin::tex2r32f                  m_noise_tex;
  ssf_rasterization_shader         m_ssf_rasterization_shader;
  domain_coverage_shader           m_domain_coverage_shader;
  ll_to_curvature_shader           m_ll_to_curvature_shader;
  weight_single_pathsurface_shader m_weight_single_pathsurface_shader;
  weight_dual_pathsurface_shader   m_weight_dual_pathsurface_shader;
  integrator_t                     m_integrator;
  RandEng&                         m_rand_eng;
  boundingbox<real_t, 3>           m_domain;
  fragment_count_shader            m_fragment_count_shader;
  combine_rasterizations_shader    m_combine_rasterizations_shader;
  // cache<size_t, path>

  //============================================================================
  // ctor
  //============================================================================
 public:
  steadification(const field<V, real_t, 2, 2>& v,
                 const boundingbox<real_t, 3>& domain, ivec2 render_resolution,
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
        m_ssf_rasterization_shader{},
        m_integrator{integration::vclibs::abs_tol      = 1e-6,
                     integration::vclibs::rel_tol      = 1e-6,
                     integration::vclibs::initial_step = 0,
                     integration::vclibs::max_step     = 0.1},
        m_rand_eng{rand_eng},
        m_domain{domain} {
    yavin::disable_multisampling();

    m_fragment_count_shader.set_projection(m_cam.projection_matrix());
    std::vector<float> noise(render_resolution(0) * render_resolution(1));
    random_uniform<float, RandEng> rand{m_rand_eng};
    boost::generate(noise, [&] { return rand(rand_eng); });
    m_noise_tex.upload_data(noise, render_resolution(0), render_resolution(1));
  }
  //============================================================================
  // methods
  //============================================================================
  auto rand() { return random_uniform<real_t, RandEng>{m_rand_eng}(); }
  //----------------------------------------------------------------------------
  template <template <typename> typename SeedcurveInterpolator>
  auto rasterize(const pathsurface_t<SeedcurveInterpolator>& mesh,
                 domain_coverage_tex_t& domain_coverage_tex) {
    return rasterize(gpu_pathsurface(mesh), domain_coverage_tex);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto rasterize(const pathsurface_gpu_t& gpu_mesh,
                 domain_coverage_tex_t&   domain_coverage_tex) {
    using namespace yavin;
    static const float nan       = 0.0f / 0.0f;
    const auto         num_frags = num_rendered_fragments(gpu_mesh);
    std::cerr << "num_frags: " << num_frags << '\n';
    rasterized_pathsurface psf_rast{
        m_render_resolution(0), m_render_resolution(1), num_frags,
        linked_list_node{{nan, nan}, {nan, nan}, {nan, nan}, nan, 0xffffffff}};
    m_ssf_rasterization_shader.bind();
    m_ssf_rasterization_shader.set_linked_list_size(psf_rast.buffer_size());
    m_ssf_rasterization_shader.set_projection(m_cam.projection_matrix());
    gl::viewport(m_cam);
    yavin::disable_depth_test();
    framebuffer fbo{domain_coverage_tex};
    fbo.bind();
    psf_rast.bind();
    gpu_mesh.draw();
    return psf_rast;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  size_t num_rendered_fragments(const pathsurface_gpu_t& gpu_mesh) {
    using namespace yavin;
    tex2r32f    col{m_render_resolution(0), m_render_resolution(1)};
    framebuffer fbo{col};
    fbo.bind();

    atomiccounterbuffer cnt{0};
    cnt.bind(1);

    m_fragment_count_shader.bind();
    yavin::gl::viewport(m_cam);
    disable_depth_test();
    gpu_mesh.draw();
    return cnt[0];
  }
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  auto pathsurface(const seedcurve_t& seedcurve, real_t t0u0, real_t t0u1,
                   real_t btau, real_t ftau, size_t seed_res,
                   real_t stepsize) const {
    using namespace VC::odeint;
    streamsurface surf{m_v, t0u0, t0u1, seedcurve, m_integrator};
    auto          mesh  = surf.discretize(seed_res, stepsize, btau, ftau);
    auto&         vprop = mesh.template add_vertex_property<vec2>("v");
    auto& curvprop = mesh.template add_vertex_property<double>("curvature");

    for (auto vertex : mesh.vertices()) {
      const auto& uv             = mesh.uv(vertex);
      const auto& integral_curve = surf.streamline_at(uv(0), 0, 0);
      curvprop[vertex]           = integral_curve.curvature(uv(1));
      // std::cerr << "curvature at uv(" << uv(0) << ", " << uv(1)
      //          << "): " << curvprop[vertex] << '\n';
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
      const pathsurface_discretization_t<SeedcurveInterpolator>& mesh) const {
    return pathsurface_gpu_t{mesh};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto gpu_pathsurface(const seedcurve_t& seedcurve, real_t t0u0, real_t t0u1,
                       real_t stepsize) const {
    return gpu_pathsurface(pathsurface(seedcurve, t0u0, t0u1, stepsize).first);
  }
  //----------------------------------------------------------------------------
  auto weight(const rasterized_pathsurface& rast) {
    tex2r32f            weight{m_render_resolution(0), m_render_resolution(1)};
    atomiccounterbuffer cnt{0};
    cnt.bind(1);
    rast.bind(0, 0, 1, 0);
    weight.bind_image_texture(2);
    m_weight_single_pathsurface_shader.dispatch(
        m_render_resolution(0) / 32.0 + 1, m_render_resolution(1) / 32.0 + 1);
    m_weight_single_pathsurface_shader.set_linked_list_size(rast.buffer_size());
    auto w = boost::accumulate(weight.download_data(), float(0));
    return w;
  }
  //----------------------------------------------------------------------------
  auto weight(const rasterized_pathsurface& rast0,
              const rasterized_pathsurface& rast1) {
    tex2r32f            weight{m_render_resolution(0), m_render_resolution(1)};
    atomiccounterbuffer cnt{0};
    rast0.bind(0, 0, 1, 0);
    rast1.bind(1, 2, 3, 1);
    cnt.bind(2);
    weight.bind_image_texture(4);
    m_weight_dual_pathsurface_shader.set_linked_list0_size(rast0.buffer_size());
    m_weight_dual_pathsurface_shader.dispatch(
        m_render_resolution(0) / 32.0 + 1, m_render_resolution(1) / 32.0 + 1);
    auto w = boost::accumulate(weight.download_data(), float(0));
    return w;
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
      // std::cerr << "kappas[" << i << "] = " << kappas[i] << '\n';
      // std::cerr << "arc_lengths[" << i << "] = " << arc_lengths[i] << '\n';
      ++i;
    }
    real_t acc_kappas = 0;
    for (size_t i = 0; i < num_integral_curves; ++i) {
      acc_kappas += kappas[i] * arc_lengths[i];
    }
    return acc_kappas / boost::accumulate(arc_lengths, real_t(0));
  }
  //============================================================================
  /// \return coverage of domain between 0 and 1; 1 meaning fully covered
  auto domain_coverage() const {
    yavin::atomiccounterbuffer covered_pixels{0};
    covered_pixels.bind(0);
    m_domain_coverage_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                      m_render_resolution(1) / 32.0 + 1);

    const auto cps   = covered_pixels.front();
    const auto nps   = m_render_resolution(0) * m_render_resolution(1);
    const auto ratio = static_cast<double>(cps) / static_cast<double>(nps);
    std::cerr << "number of covered pixels: " << cps << " / " << nps << " -> "
              << ratio << '\n';
    return ratio;
  }
  //----------------------------------------------------------------------------
  template <typename Stepsize, enable_if_arithmetic<Stepsize> = true>
  void random_domain_filling_streamsurfaces(const field<V, real_t, 2, 2>& v,
                                            Stepsize stepsize) {
    // constexpr auto domain = settings<V>::domain;
    using namespace std::filesystem;

    auto working_dir = std::string{settings<V>::name} + "/";
    if (!exists(working_dir)) { create_directory(working_dir); }
    for (const auto& entry : directory_iterator(working_dir)) { remove(entry); }

    // size_t dir_count = 1;
    // while (exists(working_dir)) {
    //  working_dir = std::string(settings<V>::name) + "_" +
    //  std::to_string(dir_count++) +
    //      "/";
    //}

    domain_coverage_tex_t domain_coverage_tex{yavin::NEAREST, yavin::REPEAT,
                                              m_render_resolution(0),
                                              m_render_resolution(1)};
    domain_coverage_tex.bind_image_texture(0);
    domain_coverage_tex.clear(0);
    std::vector<std::tuple<seedcurve_t, real_t, real_t>> seed_curves;
    // std::vector<rasterized_pathsurface>      rasterizations;

    size_t i = 0;
    while (domain_coverage() < 0.99) {
      seed_curves.emplace_back(random_seedcurve(0.1, 0.2));
      const auto& [seedcurve, t0u0, t0u1] = seed_curves.back();
      auto rast =
          rasterize(v, seedcurve, t0u0, t0u1, domain_coverage_tex, stepsize);
#if YAVIN_HAS_PNG_SUPPORT
      rast.pos.write_png(working_dir + "pos_" + std::to_string(i++) + ".png");
      domain_coverage_tex.write_png(working_dir + "coverage.png");
#endif
    }
  }
  //----------------------------------------------------------------------------
  auto random_seedcurve(real_t min_dist = 0,
                        real_t max_dist = std::numeric_limits<real_t>::max()) {
    const auto x0 = m_domain.random_point(m_rand_eng);

    auto x1 = x0;
    do {
      const auto u         = rand();
      const auto v         = rand();
      const auto theta     = u * 2 * M_PI;
      const auto phi       = std::acos(2 * v - 1);
      const auto sin_theta = std::sin(theta);
      const auto cos_theta = std::cos(theta);
      const auto sin_phi   = std::sin(phi);
      const auto cos_phi   = std::cos(phi);
      const auto r = std::cbrt(rand()) * (max_dist - min_dist) + min_dist;
      x1 = vec{r * sin_phi * cos_theta, r * sin_phi * sin_theta, r * cos_phi} +
           x0;
    } while (!m_domain.is_inside(x1));

    return std::tuple{
        seedcurve_t{{vec{x0(0), x0(1)}, 0}, {vec{x1(0), x1(1)}, 1}}, x0(2),
        x1(2)};
  }
  //----------------------------------------------------------------------------
  auto integrate_grid_edges(const field<V, real_t, 2, 2>& v,
                            const grid<real_t, 2>&        domain,
                            real_t                        stepsize) const {
    using namespace std::filesystem;
    auto working_dir = std::string{settings<V>::name} + "/";
    if (!exists(working_dir)) { create_directory(working_dir); }
    for (const auto& entry : directory_iterator(working_dir)) { remove(entry); }

    const size_t num_edges = domain.size(0) * (domain.size(1) - 1) +
                             domain.size(1) * (domain.size(0) - 1);
    size_t                                                           cnt = 0;
    std::vector<pathsurface_discretization_t<interpolation::linear>> meshes;
    for (size_t i = 0; i < domain.num_straight_edges(); ++i) {
      auto        e = domain.edge_at(i);
      seedcurve_t seedcurve{{e.first.position(), 0}, {e.second.position(), 1}};

      meshes.push_back(pathsurface(seedcurve, 0, 0, stepsize).first);
    }
    return meshes;
  }
  //----------------------------------------------------------------------------
  auto to_curvature_tex(const rasterized_pathsurface& rast) {
    tex2rgba32f v_tex{m_render_resolution(0), m_render_resolution(1)};
    rast.bind();
    v_tex.bind_image_texture(2);
    m_ll_to_curvature_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                      m_render_resolution(1) / 32.0 + 1);
    return v_tex;
  }
  //----------------------------------------------------------------------------
  auto make_domain_coverage_tex() const {
    return domain_coverage_tex_t{m_render_resolution(0),
                                 m_render_resolution(1)};
  }
  //----------------------------------------------------------------------------
  /// rast1 gets written in rast0. rast0 must have additional space to be able
  /// to hold rast1.
  void combine(rasterized_pathsurface& rast0, rasterized_pathsurface& rast1) {
    rast0.bind(0, 0, 1, 0);
    m_combine_rasterizations_shader.set_linked_list0_size(rast0.buffer_size());

    rast1.bind(1, 2, 3, 1);
    m_combine_rasterizations_shader.set_linked_list1_size(rast1.buffer_size());

    m_combine_rasterizations_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                             m_render_resolution(1) / 32.0 + 1);
    std::cerr << "counter: " << rast0.counter()[0] << '\n';
  }
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
