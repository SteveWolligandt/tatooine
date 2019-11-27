#ifndef STEADIFICATION_H
#define STEADIFICATION_H

#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/parallel_for.h>
#include <tatooine/random.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/streamsurface.h>

#include <boost/filesystem.hpp>
#include <boost/range/adaptors.hpp>
#include <cstdlib>
#include <vector>

#include "renderers.h"
#include "shaders.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real>
class steadification {
  //============================================================================
  // types
  //============================================================================
 public:
  using pathsurface_t     = mesh<Real, 3>;
  using pathsurface_gpu_t = streamsurface_renderer;
  using vec2              = vec<Real, 2>;
  using vec3              = vec<Real, 3>;
  using ivec2             = vec<size_t, 2>;
  using ivec3             = vec<size_t, 3>;
  using integrator_t = integration::vclibs::rungekutta43<double, 3>;

  struct rasterized_pathsurface {
    yavin::tex2rg<float> pos;
    yavin::tex2rg<float> v;
    yavin::tex2rg<float> uv;
    rasterized_pathsurface(size_t w, size_t h) : pos{w, h}, v{w, h}, uv{w, h} {}
  };

  //============================================================================
  // fields
  //============================================================================
 private:
  ivec2                     m_render_resolution;
  yavin::context            m_context;
  yavin::orthographiccamera m_cam;
  yavin::tex2rgb<float>     m_color_scale;
  yavin::texdepth           m_depth;
  yavin::tex2r<float>       m_noise_tex;
  v_tau_shader              m_v_tau_shader;
  integrator_t              m_integrator;

  //============================================================================
  // ctor
  //============================================================================
 public:
  template <typename RandEng = std::mt19937_64>
  steadification(const boundingbox<Real, 3>& domain,
                 ivec2     render_resolution,
                 RandEng&& rand_eng = RandEng{std::random_device{}()})
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
        m_v_tau_shader{},
        m_integrator{integration::vclibs::abs_tol      = 1e-6,
                     integration::vclibs::rel_tol      = 1e-6,
                     integration::vclibs::initial_step = 0,
                     integration::vclibs::max_step     = 0.1} {
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
                 const parameterized_line<Real, 3>& seedcurve, Real stepsize) {
    using namespace yavin;
    auto                   psf = gpu_pathsurface(v, seedcurve, stepsize);
    rasterized_pathsurface psf_rast{m_render_resolution(0),
                                    m_render_resolution(1)};
    framebuffer     fbo{psf_rast.pos, psf_rast.v, psf_rast.uv, m_depth};
    m_v_tau_shader.bind();
    m_v_tau_shader.set_projection(m_cam.projection_matrix());

    gl::viewport(m_cam.viewport());
    gl::clear_color(0,0,0,0);

    fbo.bind();
    clear_color_depth_buffer();
    psf.draw();
    psf_rast.pos.write_png("pos.png");
    psf_rast.v.write_png("v.png");
    psf_rast.uv.write_png("uv.png");
    return psf_rast;
  }
  //----------------------------------------------------------------------------
  /// \param seedcurve seedcurve in space-time
  template <typename V, typename VReal>
  auto pathsurface(const field<V, VReal, 2, 2>&       v,
                   const parameterized_line<Real, 3>& seedcurve,
                   Real                               stepsize) { spacetime_field stv{v};
    using namespace VC::odeint;
    streamsurface surf{stv,
                      0,
                      seedcurve,
                      m_integrator,
                      interpolation::linear<Real>{},
                      interpolation::hermite<Real>{}};

    auto        psf    = surf.discretize(2, stepsize, -10, 10);
    auto&       psf_v  = psf.template add_vertex_property<vec2>("v");

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
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
