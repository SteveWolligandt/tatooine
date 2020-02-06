#ifndef STEADIFICATION_H
#define STEADIFICATION_H

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
  template <typename V, template <typename> typename SeedcurveInterpolator =
                            interpolation::linear>
  using pathsurface_t =
      streamsurface<integration::vclibs::rungekutta43, SeedcurveInterpolator,
                    interpolation::hermite, V, Real, 2>;
  template <typename V, template <typename> typename SeedcurveInterpolator =
                            interpolation::linear>
  using pathsurface_discretization_t =
      hultquist_discretization<integration::vclibs::rungekutta43,
                               SeedcurveInterpolator, interpolation::hermite, V,
                               Real, 2>;
  using pathsurface_gpu_t = streamsurface_renderer;
  using vec2              = vec<Real, 2>;
  using vec3              = vec<Real, 3>;
  using ivec2             = vec<size_t, 2>;
  using ivec3             = vec<size_t, 3>;
  using integrator_t =
      integration::vclibs::rungekutta43<double, 2, interpolation::hermite>;
  using seedcurve_t = parameterized_line<double, 2, interpolation::linear>;
  using domain_coverage_tex_t = yavin::tex2r32ui;

  struct rasterized_pathsurface {
    yavin::tex2rg32f pos;
    yavin::tex2rg32f v;
    yavin::tex2rg32f uv;
    rasterized_pathsurface(size_t w, size_t h) : pos{w, h}, v{w, h}, uv{w, h} {}
  };

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
  integrator_t m_integrator;
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
  auto rasterize(const field<V, Real, 2, 2>& v, const seedcurve_t& seedcurve,
                 Real t0u0, Real t0u1,
                 domain_coverage_tex_t& domain_coverage_tex, Real stepsize) {
    using namespace yavin;
    auto                   gpu_mesh = gpu_pathsurface(v, seedcurve, t0u0, t0u1, stepsize);
    rasterized_pathsurface psf_rast{m_render_resolution(0),
                                    m_render_resolution(1)};
    framebuffer fbo{psf_rast.pos, psf_rast.v, psf_rast.uv, domain_coverage_tex,
                    m_depth};
    m_ssf_rasterization_shader.bind();
    m_ssf_rasterization_shader.set_projection(m_cam.projection_matrix());

    gl::viewport(m_cam.viewport());

    fbo.bind();
    const float nan = 0.0f / 0.0f;
    psf_rast.pos.clear(nan, nan);
    psf_rast.v.clear(nan, nan);
    psf_rast.uv.clear(nan, nan);
    clear_depth_buffer();
    gpu_mesh.draw();
    return psf_rast;
  }
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  template <typename V>
  auto pathsurface(const field<V, Real, 2, 2>& v, const seedcurve_t& seedcurve,
                   Real t0u0, Real t0u1, Real stepsize) {
    using namespace VC::odeint;
    streamsurface surf{v, t0u0, t0u1, seedcurve, m_integrator};
    auto  mesh   = surf.discretize(2, stepsize, -10, 10);
    auto& vprop = mesh.template add_vertex_property<vec2>("v");

    for (auto vertex : mesh.vertices()) {
      if (v.in_domain(mesh[vertex], mesh.uv(vertex)(1))) {
        vprop[vertex] =
            v(vec{mesh[vertex](0), mesh[vertex](1)}, mesh.uv(vertex)(1));
      } else {
        vprop[vertex] = vec<Real, 2>{0.0 / 0.0, 0.0 / 0.0};
      }
    }
    return std::pair{std::move(mesh), std::move(surf)};
  }
  //----------------------------------------------------------------------------
  template <typename V, template <typename> typename SeedcurveInterpolator>
  auto gpu_pathsurface(
      const pathsurface_discretization_t<V, SeedcurveInterpolator>& mesh) {
    return pathsurface_gpu_t{mesh};
  }
  //----------------------------------------------------------------------------
  template <typename V>
  auto gpu_pathsurface(const field<V, Real, 2, 2>& v,
                       const seedcurve_t& seedcurve, Real t0u0, Real t0u1,
                       Real stepsize) {
    return gpu_pathsurface(
        pathsurface(v, seedcurve, t0u0, t0u1, stepsize).first);
  }
  //----------------------------------------------------------------------------
  template <typename V, template <typename> typename SeedcurveInterpolator>
  auto curvature(
      const pathsurface_discretization_t<V, SeedcurveInterpolator>& mesh,
      const pathsurface_t<V, SeedcurveInterpolator>& surf) const {
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
      //std::cerr << "u = " << u << "; kappa = " << kappas.back()
      //          << "; arc length = " << arc_lengths.back() << '\n';
    }
    Real acc_kappas = 0;
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
    //constexpr auto domain = settings<V>::domain;
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
    std::vector<std::tuple<seedcurve_t, Real, Real>> seed_curves;
    // std::vector<rasterized_pathsurface>      rasterizations;

    size_t i = 0;
    while (domain_coverage() < 0.99) {
      seed_curves.emplace_back(random_seedcurve(0.1, 0.2));
      const auto& [seedcurve, t0u0, t0u1] = seed_curves.back();
      auto rast =
          rasterize(v, seedcurve, t0u0, t0u1, domain_coverage_tex, stepsize);
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

    return std::tuple{
        seedcurve_t{{vec{x0(0), x0(1)}, 0}, {vec{x1(0), x1(1)}, 1}}, x0(2),
        x1(2)};
  }

  auto rand() { return random_uniform<Real, RandEng>{m_rand_eng}(); }
  //----------------------------------------------------------------------------
  template <typename V>
  void integrate_grid_edges(const field<V, Real, 2, 2>& v,
                            const grid<Real, 2>& domain, Real stepsize) const {
    using namespace std::filesystem;
    auto working_dir = std::string{settings<V>::name} + "/";
    if (!exists(working_dir)) { create_directory(working_dir); }
    for (const auto& entry : directory_iterator(working_dir)) { remove(entry); }

    const size_t num_edges = domain.size(0) * (domain.size(1) - 1) +
                             domain.size(1) * (domain.size(0) - 1);
    size_t cnt = 0;
    std::cerr << "grid edge progress: 0 / " << num_edges << "             \r";
    std::vector<pathsurface_discretization_t<V>> meshes;
    for (size_t y = 0; y < domain.size(1); ++y) {
      for (size_t x = 0; x < domain.size(0) - 1; ++x) {
         seedcurve_t seedcurve{{domain(x, y), 0}, {domain(x + 1, y), 1}};
        //std::cerr << seedcurve.front_vertex() << " - "
        //          << seedcurve.back_vertex() << '\n';
         pathsurface_t<V> surf{v, 0, 0, seedcurve, m_integrator};
         meshes.push_back(surf.discretize(2, stepsize, -10, 10));
        std::cerr << "grid edge progress: " << ++cnt << " / " << num_edges
                  << "             \r";
      }
    }
    for (size_t x = 0; x < domain.size(0); ++x) {
      for (size_t y = 0; y < domain.size(1) - 1; ++y) {
        seedcurve_t seedcurve{{domain(x, y), 0}, {domain(x, y + 1), 1}};
        //std::cerr << seedcurve.front_vertex() << " - "
        //          << seedcurve.back_vertex() << '\n';
        pathsurface_t<V> surf{v, 0, 0, seedcurve, m_integrator};
        meshes.push_back(surf.discretize(2, stepsize, -10, 10));
        std::cerr << "grid edge progress: " << ++cnt << " / " << num_edges
                  << "             \r";
      }
    }
    std::cerr << '\n';
    cnt = 0;
    for (const auto& mesh : meshes) {
      mesh.write_vtk("mesh_" + std::to_string(cnt++) + ".vtk");
    }
  }
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
