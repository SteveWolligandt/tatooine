#ifndef STEADIFICATION_H
#define STEADIFICATION_H

#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/parallel_for.h>
#include <tatooine/random.h>
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
 public:
  using ribbon_t     = tatooine::mesh<Real, 2>;
  using ribbon_gpu_t = StreamsurfaceRenderer;

  struct rendered_surface {
    tex2rgba<float>   vector_tex;
    tex2rgba<uint8_t> time_tex;
    rendered_surface(size_t w, size_t h) : vector_tex{w, h}, time_tex{w, h} {}
  };

 private:
  tatooine::vec<size_t, 2> m_render_resolution;
  glfw_window              m_w;
  orthographiccamera       m_cam;
  boundingbox<Real, 3>     m_domain;
  tex2rgb<float>           m_color_scale;
  texdepth                 m_depth;
  tex2r<float>             m_noise_tex;

  LineShader line_shader;

 public:
  //============================================================================
  steadification(const tatooine::boundingbox<Real, 3>& domain,
                 tatooine::vec<size_t, 2> window_resolution,
                 tatooine::vec<size_t, 2> render_resolution,
                 size_t _seed_res, Real _stepsize)
      : m_domain{domain},
        m_render_resolution{render_resolution},
        w("steadification", window_resolution(0), window_resolution(1)),
        cam(domain.min(0), domain.max(0), domain.min(1), domain.max(1), -1000,
            1000, render_resolution(0), render_resolution(1)),

        color_scale{yavin::LINEAR, yavin::CLAMP_TO_EDGE, "color_scale.png"},
        depth{yavin::NEAREST, yavin::CLAMP_TO_EDGE, render_resolution(0),
              render_resolution(1)},
        noise_tex{yavin::LINEAR, yavin::REPEAT, render_resolution(0),
                  render_resolution(1)} {
    yavin::disable_multisampling();

    std::vector<float>    noise(render_resolution(0), render_resolution(1));
    random_uniform<float> rand{0, 1};
    boost::generate(noise, [&rand] { return rand(); });
    noise_tex.upload_data(noise.data(), render_resolution(0),
                          render_resolution(1));
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
  template <typename V>
  float evaluate(const V& v, const solution_t& sol) {
    to_linked_list(v, sol);
    show_current(sol);
    // std::this_thread::sleep_for(std::chrono::seconds{2});
    return weight();
  }

  //----------------------------------------------------------------------------
  template <typename V>
  auto pathsurface(const V& v, const edge_t& e, Real stepsize) {
    using namespace VC::odeint;
    using vec2 = tatooine::vec<Real, 2>;
    tatooine::streamsurface ssf{
        v,
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
    auto&       mesh_vf = ribbon.template add_vertex_property<vec2>("v");

    for (auto v : ribbon.vertices()) {
      if (v.in_domain(ribbon[v], t0 + mesh_uv[v](1))) {
        mesh_vf[v] = v(ribbon[v], t0 + mesh_uv[v](1));
      } else {
        mesh_vf[v] = tatooine::vec<Real, 2>{0.0 / 0.0, 0.0 / 0.0};
      }
    }

    return ribbon;
  }

  //----------------------------------------------------------------------------
  template <typename V>
  auto pathsurface(const V& v, const edge_t& e) {
    return pathsurface(v, e, stepsize);
  }
  //----------------------------------------------------------------------------
  template <typename V>
  ribbon_gpu_t gpu_pathsurface(const V& v, const edge_t& e,
                                   Real stepsize) {
    return StreamsurfaceRenderer(pathsurface(v, e, stepsize));
  }

  //----------------------------------------------------------------------------
  template <typename V>
  ribbon_gpu_t gpu_pathsurface(const V& v, const edge_t& e) {
    return gpu_pathsurface(v, e, stepsize);
  }

  //--------------------------------------------------------------------------
  template <typename V, typename RandEng>
  auto calc(const V& v, size_t _num_its, size_t _seedcurve_length,
            const std::string& path, Real desired_coverage, RandEng&& eng,
            const std::vector<listener_t*>& listeners = {}) {
  }

};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
