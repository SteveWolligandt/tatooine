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

#include "renderers.h"
#include "shaders.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, typename RandEng>
class steadification {
  //============================================================================
  // types
  //============================================================================
 public:
  using pathsurface_t         = simple_tri_mesh<Real, 3>;
  using pathsurface_gpu_t     = streamsurface_renderer;
  using vec2                  = vec<Real, 2>;
  using vec3                  = vec<Real, 3>;
  using ivec2                 = vec<size_t, 2>;
  using ivec3                 = vec<size_t, 3>;
  using integrator_t          = integration::vclibs::rungekutta43<double, 3>;
  using domain_coverage_tex_t = yavin::tex2rui32;

  struct rasterized_pathsurface {
    yavin::tex2rgf pos;
    yavin::tex2rgf v;
    yavin::tex2rgf uv;
    rasterized_pathsurface(size_t w, size_t h) : pos{w, h}, v{w, h}, uv{w, h} {}
  };

  //============================================================================
  // fields
  //============================================================================
 private:
  ivec2                     m_render_resolution;
  yavin::egl::context       m_context;
  yavin::orthographiccamera m_cam;
  yavin::tex2rgbf           m_color_scale;
  yavin::texdepth           m_depth;
  yavin::tex2rf             m_noise_tex;
  ssf_rasterization_shader  m_ssf_rasterization_shader;
  domain_coverage_shader    m_domain_coverage_shader;
  integrator_t              m_integrator;
  RandEng&                  m_rand_eng;

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
        m_rand_eng{rand_eng} {
    yavin::disable_multisampling();

    std::vector<float> noise(render_resolution(0) * render_resolution(1));
    random_uniform     rand{0.0f, 1.0f};
    boost::generate(noise, [&] { return rand(rand_eng); });
    m_noise_tex.upload_data(noise, render_resolution(0), render_resolution(1));
  }
  //============================================================================
  // methods
  //============================================================================
  template <typename V, typename VReal>
  auto rasterize(const field<V, VReal, 2, 2>&       v,
                 const parameterized_line<Real, 3>& seedcurve,
                 domain_coverage_tex_t& domain_coverage_tex, Real stepsize) {
    using namespace yavin;
    auto                   psf = gpu_pathsurface(v, seedcurve, stepsize);
    rasterized_pathsurface psf_rast{m_render_resolution(0),
                                    m_render_resolution(1)};
    framebuffer fbo{psf_rast.pos, psf_rast.v, psf_rast.uv, domain_coverage_tex,
                    m_depth};
    m_ssf_rasterization_shader.bind();
    m_ssf_rasterization_shader.set_projection(m_cam.projection_matrix());

    gl::viewport(m_cam.viewport());

    fbo.bind();
    const float nan = 0.0f/ 0.0f;
    psf_rast.pos.clear(nan, nan);
    psf_rast.v.clear(nan, nan);
    psf_rast.uv.clear(nan, nan);
    clear_depth_buffer();
    psf.draw();
    return psf_rast;
  }
  //----------------------------------------------------------------------------
  template <typename... Rasterizations>
  auto rasterize(const Rasterizations... rasterizations) {
    rasterized_pathsurface combined{m_render_resolution(0),
                                    m_render_resolution(1)};
  }
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  template <typename V, typename VReal>
  auto pathsurface(const field<V, VReal, 2, 2>&       v,
                   const parameterized_line<Real, 3>& seedcurve,
                   Real                               stepsize) {
    spacetime_field stv{v};
    using namespace VC::odeint;
    streamsurface surf{stv,
                       0,
                       seedcurve,
                       m_integrator,
                       interpolation::linear<Real>{},
                       interpolation::hermite<Real>{}};

    auto  psf   = surf.discretize(2, stepsize, -10, 10);
    auto& psf_v = psf.template add_vertex_property<vec2>("v");

    for (auto vertex : psf.vertices()) {
      if (stv.in_domain(psf[vertex], 0)) {
        psf_v[vertex] = v(vec{psf[vertex](0), psf[vertex](1)}, psf[vertex](2));
      } else {
        psf_v[vertex] = vec{0.0 / 0.0, 0.0 / 0.0};
      }
    }

    return psf;
  }
  //----------------------------------------------------------------------------
  auto gpu_pathsurface(const pathsurface_t& psf) {
    return pathsurface_gpu_t{psf};
  }
  //----------------------------------------------------------------------------
  template <typename V, typename VReal>
  auto gpu_pathsurface(const field<V, VReal, 2, 2>&       v,
                       const parameterized_line<Real, 3>& seedcurve,
                       Real                               stepsize) {
    return gpu_pathsurface(pathsurface(v, seedcurve, stepsize));
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
  template <typename V, typename VReal, typename Stepsize,
            enable_if_arithmetic<Stepsize> = true>
  void random_domain_filling_streamsurfaces(const field<V, VReal, 2, 2>& v,
                                            Stepsize stepsize) {
    static constexpr auto domain = settings<V>::domain;
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
    //for (; i < 3;) {
    while (domain_coverage() < 0.9) {
      seed_curves.emplace_back(
          parameterized_line<Real, 3>{{domain.random_point(m_rand_eng), 0},
                                      {domain.random_point(m_rand_eng), 1}});
      // rasterizations.push_back(rasterize(v, seed_curves.back(), 0.1));
      auto rast =
          rasterize(v, seed_curves.back(), domain_coverage_tex, stepsize);
      rast.pos.write_png(working_dir + "pos_" + std::to_string(i++) + ".png");
      domain_coverage_tex.write_png(working_dir + "coverage.png");
    }
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
