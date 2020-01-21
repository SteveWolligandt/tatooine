#ifndef TATOOINE_STEADIFICATION_STEADIFICATION_H
#define TATOOINE_STEADIFICATION_STEADIFICATION_H

#include <tatooine/for_loop.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/random.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/streamsurface.h>

#include <boost/filesystem.hpp>
#include <boost/range/adaptors.hpp>
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

template <typename Real, typename RandEng>
class steadification {
  //============================================================================
  // types
  //============================================================================
 public:
  template <template <typename, size_t> typename Integrator,
            template <typename> typename SeedcurveInterpolator,
            template <typename> typename StreamlineInterpolator, typename V>
  using pathsurface_t =
      hultquist_discretization<Integrator, SeedcurveInterpolator,
                               StreamlineInterpolator, V, Real, 3>;
  using pathsurface_gpu_t     = streamsurface_renderer;
  using vec2                  = vec<Real, 2>;
  using vec3                  = vec<Real, 3>;
  using ivec2                 = vec<size_t, 2>;
  using ivec3                 = vec<size_t, 3>;
  using integrator_t          = integration::vclibs::rungekutta43<double, 3>;
  using domain_coverage_tex_t = yavin::tex2r32ui;

  struct linked_list_node {
    vec<float, 2> pos;
    vec<float, 2> v;
    vec<float, 2> uv;
    unsigned int  next;
  };
  using rasterized_pathsurface = linked_list_texture<linked_list_node>;

  //============================================================================
  // members
  //============================================================================
 private:
  ivec2                     m_render_resolution;
  yavin::context            m_context;
  yavin::orthographiccamera m_cam;
  yavin::tex2rgb32f         m_color_scale;
  yavin::texdepth           m_depth;
  yavin::tex2r32f           m_noise_tex;
  ssf_rasterization_shader  m_ssf_rasterization_shader;
  domain_coverage_shader    m_domain_coverage_shader;
  ll_to_pos_shader          m_ll_to_pos_shader;
  integrator_t              m_integrator;
  RandEng&                  m_rand_eng;
  boundingbox<Real, 3>      m_domain;

  //============================================================================
  // ctor
  //============================================================================
 public:
  steadification(const boundingbox<Real, 3>& domain, ivec2 render_resolution,
                 RandEng& rand_eng)
      : m_render_resolution{render_resolution},
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

    std::vector<float> noise(render_resolution(0) * render_resolution(1));
    random_uniform<float, RandEng> rand{m_rand_eng};
    boost::generate(noise, [&] { return rand(rand_eng); });
    m_noise_tex.upload_data(noise, render_resolution(0), render_resolution(1));
  }
  //============================================================================
  // methods
  //============================================================================
  template <typename V>
  auto rasterize(const field<V, Real, 2, 2>&       v,
                 const parameterized_line<Real, 3>& seedcurve, Real stepsize,
                 domain_coverage_tex_t& domain_coverage_tex) {
    return rasterize(pathsurface(v, seedcurve, stepsize).first, domain_coverage_tex);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <template <typename, size_t> typename Integrator,
            template <typename> typename SeedcurveInterpolator,
            template <typename> typename StreamlineInterpolator, typename V>
  auto rasterize(const pathsurface_t<Integrator, SeedcurveInterpolator,
                                     StreamlineInterpolator, V>& mesh,
                 domain_coverage_tex_t& domain_coverage_tex) {
    return rasterize(gpu_pathsurface(mesh), domain_coverage_tex);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto rasterize(const pathsurface_gpu_t& gpu_mesh,
                 domain_coverage_tex_t& domain_coverage_tex) {
    using namespace yavin;
    static const float nan = 0.0f / 0.0f;
    rasterized_pathsurface psf_rast{
        m_render_resolution(0), m_render_resolution(1),
        linked_list_node{{nan, nan}, {nan, nan}, {nan, nan}, 0xffffffff}};
    m_ssf_rasterization_shader.bind();
    m_ssf_rasterization_shader.set_linked_list_size(psf_rast.buffer_size());
    m_ssf_rasterization_shader.set_projection(m_cam.projection_matrix());
    gl::viewport(m_cam.viewport());
    yavin::disable_depth_test();
    framebuffer fbo{domain_coverage_tex};
    fbo.bind();
    psf_rast.bind();
    gpu_mesh.draw();
    return psf_rast;
  }
  //----------------------------------------------------------------------------
  //template <typename... Rasterizations>
  //auto rasterize(const Rasterizations... rasterizations) {
  //  rasterized_pathsurface combined{m_render_resolution(0),
  //                                  m_render_resolution(1)};
  //}
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  template <typename V>
  auto pathsurface(const field<V, Real, 2, 2>&        v,
                   const parameterized_line<Real, 3>& seedcurve, Real stepsize,
                   size_t seedres) {
    spacetime_field stv{v};
    using namespace VC::odeint;
    streamsurface surf{stv,
                       0,
                       seedcurve,
                       m_integrator,
                       interpolation::linear<Real>{},
                       interpolation::hermite<Real>{}};

    auto  mesh  = surf.discretize(seedres, stepsize, -10, 10);
    auto& vprop = mesh.template add_vertex_property<vec2>("v");

    for (auto vertex : mesh.vertices()) {
      if (stv.in_domain(mesh[vertex], 0)) {
        vprop[vertex] = v(vec{mesh[vertex](0), mesh[vertex](1)}, mesh[vertex](2));
      } else {
        vprop[vertex] = vec{0.0 / 0.0, 0.0 / 0.0};
      }
    }
    return std::pair{std::move(mesh), std::move(surf)};
  }
  //----------------------------------------------------------------------------
  template <template <typename, size_t> typename Integrator,
            template <typename> typename SeedcurveInterpolator,
            template <typename> typename StreamlineInterpolator, typename V>
  auto gpu_pathsurface(const pathsurface_t<Integrator, SeedcurveInterpolator,
                                           StreamlineInterpolator, V>& mesh) {
    return pathsurface_gpu_t{mesh};
  }
  //----------------------------------------------------------------------------
  template <typename V>
  auto gpu_pathsurface(const field<V, Real, 2, 2>&       v,
                       const parameterized_line<Real, 3>& seedcurve,
                       Real                               stepsize) {
    return gpu_pathsurface(pathsurface(v, seedcurve, stepsize).first);
  }
  //----------------------------------------------------------------------------
  template <template <typename, size_t> typename Integrator,
            template <typename> typename SeedcurveInterpolator,
            template <typename> typename StreamlineInterpolator, typename VSurf>
  auto curvature(
      const pathsurface_t<Integrator, SeedcurveInterpolator,
                          StreamlineInterpolator, VSurf>&          mesh,
      const streamsurface<Integrator, SeedcurveInterpolator,
                          StreamlineInterpolator, VSurf, Real, 3>& surf) const {
    std::set<Real> us;
    for (auto v : mesh.vertices()) { us.insert(mesh.uv(v)(0)); }
    const auto num_integral_curves = us.size();

    std::vector<Real> kappas, arc_lengths;
    arc_lengths.reserve(num_integral_curves);
    kappas.reserve(num_integral_curves);
    for (auto u : us) {
      const auto& integral_curve = surf.streamline_at(u, 0, 0);
      kappas.push_back(integral_curve.integrated_curvature());
      arc_lengths.push_back(integral_curve.arc_length());
      std::cerr << "u = " << u << "; kappa = " << kappas.back()
                << "; arc length = " << arc_lengths.back() << '\n';
    }
    Real       acc_kappas          = 0;
    for (size_t i = 0; i < num_integral_curves; ++i) {
      acc_kappas += kappas[i] * arc_lengths[i];
    }
    return acc_kappas / boost::accumulate(arc_lengths, Real(0));
  }
  //============================================================================
  /// \return coverage of domain between 0 and 1; 1 meaning fully covered
  auto domain_coverage() {
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
  //============================================================================
  template <typename V, typename Stepsize,
            enable_if_arithmetic<Stepsize> = true>
  void random_domain_filling_streamsurfaces(const field<V, Real, 2, 2>& v,
                                            Stepsize stepsize) {
    constexpr auto domain = settings<V>::domain;
    using namespace std::filesystem;

    auto working_dir = std::string{settings<V>::name} + "/";
    if (!exists(working_dir)) { create_directory(working_dir); }
    for (const auto& entry : directory_iterator(working_dir)) {
      remove(entry);
    }

    // size_t dir_count = 1;
    // while (exists(working_dir)) {
    //  working_dir = std::string(settings<V>::name) + "_" + std::to_string(dir_count++) +
    //      "/";
    //}

    domain_coverage_tex_t domain_coverage_tex{yavin::NEAREST, yavin::REPEAT,
                                              m_render_resolution(0),
                                              m_render_resolution(1)};
    domain_coverage_tex.bind_image_texture(0);
    domain_coverage_tex.clear(0);
    std::vector<parameterized_line<Real, 3>> seed_curves;
    // std::vector<rasterized_pathsurface>      rasterizations;

    size_t i = 0;
    while (domain_coverage() < 0.99) {
      seed_curves.emplace_back(random_seedcurve(0.1, 0.2));
      // rasterizations.push_back(rasterize(v, seed_curves.back(), 0.1));
      auto rast =
          rasterize(v, seed_curves.back(), domain_coverage_tex, stepsize);
      rast.pos.write_png(working_dir + "pos_" + std::to_string(i++) + ".png");
      domain_coverage_tex.write_png(working_dir + "coverage.png");
    }
  }
  //----------------------------------------------------------------------------
  auto random_seedcurve(Real min_dist = 0,
                        Real max_dist = std::numeric_limits<Real>::max()) {
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

    return parameterized_line<Real, 3>{{x0, 0}, {x1, 1}};
  }
  //----------------------------------------------------------------------------
  auto rand() { return random_uniform<Real, RandEng>{m_rand_eng}(); }
  //----------------------------------------------------------------------------
  auto to_pos_tex(rasterized_pathsurface& r) {
    tex2rg<float> v_tex{m_render_resolution(0), m_render_resolution(1)};
    v_tex.bind_image_texture(2);
    r.bind();
    m_ll_to_pos_shader.dispatch(32, 32);
    return v_tex;
  }
  //----------------------------------------------------------------------------
  auto make_domain_coverage_tex() const {
    return domain_coverage_tex_t{m_render_resolution(0),
                                 m_render_resolution(1)};
  }
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
