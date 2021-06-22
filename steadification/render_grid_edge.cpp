#include <tatooine/chrono.h>
#include <tatooine/for_loop.h>

#include <cstdlib>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/random.h>
#include <tatooine/streamsurface.h>

#include <vector>
#include <tatooine/gl/shaderstoragebuffer.h>

#include "renderers.h"
#include "settings.h"
#include "shaders.h"
#include "pathsurface.h"

//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename V, typename RandEng>
class renderer {
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
  using pathsurface_gpu_t    = streamsurface_renderer;
  using vec2                 = vec<real_t, 2>;
  using vec3                 = vec<real_t, 3>;
  using ivec2                = vec<size_t, 2>;
  using ivec3                = vec<size_t, 3>;
  using grid_t               = grid<real_t, 2>;
  using grid_edge_iterator_t = typename grid_t::edge_iterator;
  inline static const float nanf = 0.0f / 0.0f;
  struct node {
    vec<GLfloat, 2> v;
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
  orthographiccamera          m_cam;
  gl::tex2rgb32f                  m_color_scale;
  gl::tex2r32f                    m_noise_tex;
  gl::tex2r32f                    m_fbotex;
  gl::tex2rgba32f                 m_seedcurvetex;
  gl::framebuffer                 m_fbo;
  ssf_rasterization_shader           m_ssf_rasterization_shader;
  tex_rasterization_to_buffer_shader m_tex_rasterization_to_buffer_shader;
  lic_shader                         m_lic_shader;
  ll_to_v_shader                     m_result_to_v_tex_shader;
  RandEng&                           m_rand_eng;
  boundingbox<real_t, 2>             m_domain;
  combine_rasterizations_shader      m_combine_rasterizations_shader;
  seedcurve_shader                   m_seedcurve_shader;
 public:
  gl::tex2rgba32f  m_front_v_t_t0;
  gl::tex2r32f     m_front_curvature;
  gl::tex2rg32ui   m_front_renderindex_layer;
  gl::texdepth32ui m_front_depth;
  gl::framebuffer  m_front_fbo;

  gl::tex2rgba32f  m_back_v_t_t0;
  gl::tex2r32f     m_back_curvature;
  gl::tex2rg32ui   m_back_renderindex_layer;
  gl::texdepth32ui m_back_depth;
  gl::framebuffer  m_back_fbo;

  gl::tex2rgba32f m_lic_tex;
  gl::tex2rgba32f m_curvature_lic_tex;
  gl::tex2rgba32f m_color_lic_tex;
  gl::tex2rgba32f m_v_tex;

  gl::shaderstoragebuffer<node>    m_result_rasterization;
  gl::shaderstoragebuffer<node>    m_working_rasterization;
  gl::shaderstoragebuffer<GLuint>  m_result_list_size;
  gl::shaderstoragebuffer<GLuint>  m_working_list_size;

  gl::atomiccounterbuffer m_num_overall_covered_pixels{0};
  gl::atomiccounterbuffer m_num_newly_covered_pixels{0};

  //============================================================================
  // ctor
  //============================================================================
 public:
  renderer(const field<V, real_t, 2, 2>& v,
                 const boundingbox<real_t, 2>& domain, ivec2 render_resolution,
                 RandEng& rand_eng)
      : m_v{v.as_derived()},
        m_render_resolution{render_resolution},
        m_cam{static_cast<float>(domain.min(0)),
              static_cast<float>(domain.max(0)),
              static_cast<float>(domain.min(1)),
              static_cast<float>(domain.max(1)),
              -100000,
              100000,
              render_resolution(0),
              render_resolution(1)},
        m_color_scale{gl::LINEAR, gl::CLAMP_TO_EDGE, "color_scale.png"},
        m_noise_tex{gl::LINEAR, gl::REPEAT, render_resolution(0),
                    render_resolution(1)},
        m_fbotex{render_resolution(0), render_resolution(1)},
        m_seedcurvetex{render_resolution(0), render_resolution(1)},
        m_fbo{m_fbotex},
        m_rand_eng{rand_eng},
        m_domain{domain},

        m_front_v_t_t0{m_render_resolution(0), m_render_resolution(1)},
        m_front_curvature{m_render_resolution(0), m_render_resolution(1)},
        m_front_renderindex_layer{m_render_resolution(0),
                                  m_render_resolution(1)},
        m_front_depth{m_render_resolution(0), m_render_resolution(1)},
        m_front_fbo{m_front_v_t_t0, m_front_curvature,
                    m_front_renderindex_layer, m_front_depth},

        m_back_v_t_t0{m_render_resolution(0), m_render_resolution(1)},
        m_back_curvature{m_render_resolution(0), m_render_resolution(1)},
        m_back_renderindex_layer{m_render_resolution(0),
                                 m_render_resolution(1)},
        m_back_depth{m_render_resolution(0), m_render_resolution(1)},
        m_back_fbo{m_back_v_t_t0, m_back_curvature, m_back_renderindex_layer,
                   m_back_depth},
        m_lic_tex{m_render_resolution(0), m_render_resolution(1)},
        m_curvature_lic_tex{m_render_resolution(0), m_render_resolution(1)},
        m_color_lic_tex{m_render_resolution(0), m_render_resolution(1)},
        m_v_tex{m_render_resolution(0), m_render_resolution(1)},
        m_result_rasterization(
            2 * m_render_resolution(0) * m_render_resolution(1),
            {{nanf, nanf}, nanf, nanf, nanf, 0, 0, nanf}),
        m_working_rasterization(
            2 * m_render_resolution(0) * m_render_resolution(1),
            {{nanf, nanf}, nanf, nanf, nanf, 0, 0, nanf}),
        m_result_list_size(m_render_resolution(0) * m_render_resolution(1), 0),
        m_working_list_size(m_render_resolution(0) * m_render_resolution(1), 0),
        m_num_overall_covered_pixels{0},
        m_num_newly_covered_pixels{0} {
          gl::disable_multisampling();

    m_seedcurve_shader.set_projection(m_cam.projection_matrix());
    m_ssf_rasterization_shader.set_projection(m_cam.projection_matrix());
    m_ssf_rasterization_shader.set_width(m_render_resolution(0));
    const auto noise_data = random::uniform_vector<float>(
        render_resolution(0) * render_resolution(1), 0.0f, 1.0f, m_rand_eng);
    m_noise_tex.upload_data(noise_data, m_render_resolution(0),
                            m_render_resolution(1));

    m_front_v_t_t0.bind_image_texture(0);
    m_front_curvature.bind_image_texture(1);
    m_front_renderindex_layer.bind_image_texture(2);
    m_back_v_t_t0.bind_image_texture(3);
    m_back_curvature.bind_image_texture(4);
    m_back_renderindex_layer.bind_image_texture(5);
    m_lic_tex.bind_image_texture(6);
    m_v_tex.bind_image_texture(7);
    m_result_rasterization.bind(0);
    m_working_rasterization.bind(1);
    m_result_list_size.bind(2);
    m_working_list_size.bind(3);

    m_result_list_size.upload_data(0);
    m_num_overall_covered_pixels.bind(0);
    m_num_newly_covered_pixels.bind(1);
    m_v_tex.bind(0);
    m_noise_tex.bind(1);
    m_color_scale.bind(2);
    m_ssf_rasterization_shader.set_width(m_render_resolution(0));
    m_combine_rasterizations_shader.set_resolution(m_render_resolution(0),
                                                   m_render_resolution(1));
    m_lic_shader.set_v_tex_bind_point(0);
    m_lic_shader.set_noise_tex_bind_point(1);
    m_lic_shader.set_color_scale_bind_point(2);
    gl::viewport(m_cam);
    gl::enable_depth_test();
  }
  //============================================================================
  // methods
  //============================================================================
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

    m_working_list_size.upload_data(0);
    m_front_fbo.bind();
    gl::clear_color_depth_buffer();
    gl::depth_func_less();
    m_ssf_rasterization_shader.set_count(true);
    gl::barrier();
    gpu_mesh.draw();
    gl::barrier();

    m_back_fbo.bind();
    gl::clear_color_depth_buffer();
    m_back_depth.clear(1e5);
    gl::depth_func_greater();
    m_ssf_rasterization_shader.set_count(false);
    gl::barrier();
    gpu_mesh.draw();
    gl::barrier();
  }
  //----------------------------------------------------------------------------
  void to_shaderstoragebuffer() {
    gl::barrier();
    m_tex_rasterization_to_buffer_shader.dispatch(
        m_render_resolution(0) / 32.0 + 1, m_render_resolution(1) / 32.0 + 1);
    gl::barrier();
  }
  //----------------------------------------------------------------------------
  void result_to_lic_tex(const grid_t& domain, const real_t min_t,
                         const real_t max_t) {
    const size_t num_samples = 20;
    const real_t stepsize =
        (domain.dimension(0).back() - domain.dimension(0).front()) /
        (m_render_resolution(0) * 2);
    result_to_v_tex();

    m_curvature_lic_tex.bind_image_texture(5);
    m_color_lic_tex.bind_image_texture(7);
    m_lic_tex.clear(1, 1, 1, 0);
    m_curvature_lic_tex.clear(1, 1, 1, 0);
    m_color_lic_tex.clear(1, 1, 1, 0);
    m_lic_shader.set_domain_min(domain.front(0), domain.front(1));
    m_lic_shader.set_domain_max(domain.back(0), domain.back(1));
    m_lic_shader.set_min_t(min_t);
    m_lic_shader.set_max_t(max_t);
    m_lic_shader.set_num_samples(num_samples);
    m_lic_shader.set_stepsize(stepsize);
    m_lic_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                          m_render_resolution(1) / 32.0 + 1);
    m_v_tex.bind_image_texture(7);
    m_back_renderindex_layer.bind_image_texture(5);
  }
  //----------------------------------------------------------------------------
  auto result_to_v_tex() {
    gl::barrier();
    m_result_to_v_tex_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                      m_render_resolution(1) / 32.0 + 1);
    gl::barrier();
  }
  //----------------------------------------------------------------------------
  /// rast1 gets written in rast0. rast0 must have additional space to be able
  /// to hold rast1.
  void combine() {
    m_combine_rasterizations_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                             m_render_resolution(1) / 32.0 + 1);
    gl::barrier();
  }
  void render_grid(gl::tex2rgba32f& tex, const grid_t& domain) {
    gl::disable_depth_test();
    gl::framebuffer fbo{tex};
    fbo.bind();
    m_seedcurve_shader.bind();
    std::vector<line<real_t, 3>> domain_edges;

    for (auto edge : domain.edges()) {
      auto [v0, v1] = edge;
      auto x0       = v0.position();
      auto x1       = v1.position();
      domain_edges.push_back(
          line<real_t, 3>{vec<real_t, 3>{x0(0), x0(1), 0},
                          vec<real_t, 3>{x1(0), x1(1), 0}});
    }
    //for (auto x : domain.dimension(0)) {
    //  domain_edges.push_back(
    //      line<real_t, 3>{vec<real_t, 3>{x, domain.dimension(1).front(), 0},
    //                      vec<real_t, 3>{x, domain.dimension(1).back(), 0}});
    //}
    //for (auto y : domain.dimension(1)) {
    //  domain_edges.push_back(
    //      line<real_t, 3>{vec<real_t, 3>{domain.dimension(0).front(), y, 0},
    //                      vec<real_t, 3>{domain.dimension(0).back(), y, 0}});
    //}

    auto domain_edges_gpu = line_renderer(domain_edges);
    m_seedcurve_shader.use_color_scale(false);
    m_seedcurve_shader.set_color(0.0f, 0.0f, 0.0f, 0.8f);
    gl::line_width(2);
    gl::enable_blending();
    gl::blend_func_alpha();
    domain_edges_gpu.draw();
    gl::enable_depth_test();
    gl::disable_blending();
  }
  void render_seedcurve(gl::tex2rgba32f&                 tex,
                         line<real_t, 2>& seed,
                         GLfloat t0, GLfloat min_t, GLfloat max_t) {
    gl::disable_depth_test();
    gl::framebuffer fbo{tex};
    fbo.bind();
    m_seedcurve_shader.bind();

    auto renderer = line_renderer(std::vector{seed});
    m_seedcurve_shader.set_min_t(min_t);
    m_seedcurve_shader.set_max_t(max_t);
    m_seedcurve_shader.set_t0(t0);
    //m_seedcurve_shader.use_color_scale(true);
    m_seedcurve_shader.set_color(1.0f, 0.0f, 0.0f, 1.0f);
    gl::line_width(4);
    renderer.draw();
    gl::enable_depth_test();
  }
  //----------------------------------------------------------------------------
  auto render(const grid_t& domain, const typename grid_t::edge& edge,
              const real_t min_t, const real_t max_t, const real_t t0, const real_t btau,
              const real_t ftau, const size_t seed_res, const real_t stepsize) {
    using namespace std::string_literals;

    simple_tri_mesh<real_t, 2> mesh =
        pathsurface(m_v, edge, t0, btau, ftau, seed_res, stepsize)
            .first;
    rasterize(pathsurface_gpu_t{mesh, t0}, 0, 0);
    combine();

    result_to_lic_tex(domain, min_t, max_t);
    render_grid(m_lic_tex, domain);
    render_grid(m_color_lic_tex, domain);
    const auto [v0, v1] = edge;
    line seed{v0.position(), v1.position()};
    render_seedcurve(m_lic_tex, seed, t0, min_t, max_t);
    render_seedcurve(m_color_lic_tex, seed, t0, min_t, max_t);
    m_lic_tex.write_png("single_pathsurface.png");
    m_color_lic_tex.write_png("single_pathsurface_colored.png");
  }
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================

int main(int , char** argv) {
  gl::context ctx{4, 5};
  using namespace tatooine;
  using namespace steadification;
  using namespace numerical;
  using V = doublegyre<double>;
  V v;
  constexpr auto  dom = settings<V>::domain;
  std::mt19937_64 randeng{1234};
  renderer  s(v, dom, vec<size_t, 2>{500, 250}, randeng);
  grid            domain{linspace{dom.min(0), dom.max(0), 21ul},
                         linspace{dom.min(1), dom.max(1), 11ul}};
  auto edge_it = domain.edges().begin();
  for (size_t i = 0; i < atoi(argv[1]); ++i) { ++edge_it; }
  s.render(domain, *edge_it, 0, 10, 5, -5, 5, 4, 0.05);
}
