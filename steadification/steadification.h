#ifndef TATOOINE_STEADIFICATION_STEADIFICATION_H
#define TATOOINE_STEADIFICATION_STEADIFICATION_H

#include <tatooine/chrono.h>
#include <tatooine/for_loop.h>

#include <cstdlib>
//#include <tatooine/gpu/reduce.h>
#include <tatooine/html.h>
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
#include "settings.h"
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
  yavin::orthographiccamera          m_cam;
  yavin::tex2rgb32f                  m_color_scale;
  yavin::tex2r32f                    m_noise_tex;
  yavin::tex2r32f                    m_fbotex;
  yavin::tex2rgba32f                 m_seedcurvetex;
  yavin::framebuffer                 m_fbo;
  ssf_rasterization_shader           m_ssf_rasterization_shader;
  tex_rasterization_to_buffer_shader m_tex_rasterization_to_buffer_shader;
  lic_shader                         m_lic_shader;
  ll_to_v_shader                     m_ll_to_v_shader;
  ll_to_curvature_shader             m_to_curvature_shader;
  weight_dual_pathsurface_shader     m_weight_dual_pathsurface_shader;
  RandEng&                           m_rand_eng;
  boundingbox<real_t, 2>             m_domain;
  combine_rasterizations_shader      m_combine_rasterizations_shader;
  seedcurve_shader                   m_seedcurve_shader;
  // coverage_shader                  m_coverage_shader;
  // dual_coverage_shader             m_dual_coverage_shader;
 public:
  yavin::tex2rgba32f  m_front_v_t_t0;
  yavin::tex2r32f     m_front_curvature;
  yavin::tex2rg32ui   m_front_renderindex_layer;
  yavin::texdepth32ui m_front_depth;
  yavin::framebuffer  m_front_fbo;

  yavin::tex2rgba32f  m_back_v_t_t0;
  yavin::tex2r32f     m_back_curvature;
  yavin::tex2rg32ui   m_back_renderindex_layer;
  yavin::texdepth32ui m_back_depth;
  yavin::framebuffer  m_back_fbo;

  yavin::tex2rgba32f m_lic_tex;
  yavin::tex2rgba32f m_color_lic_tex;
  yavin::tex2rgba32f m_v_tex;

  yavin::shaderstoragebuffer<node>    m_result_rasterization;
  yavin::shaderstoragebuffer<node>    m_working_rasterization;
  yavin::shaderstoragebuffer<GLuint>  m_result_list_size;
  yavin::shaderstoragebuffer<GLuint>  m_working_list_size;
  yavin::shaderstoragebuffer<GLfloat> m_weight_buffer;

  yavin::atomiccounterbuffer m_num_overall_covered_pixels{0};
  yavin::atomiccounterbuffer m_num_newly_covered_pixels{0};

  //============================================================================
  // ctor
  //============================================================================
 public:
  steadification(const field<V, real_t, 2, 2>& v,
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
        m_color_scale{yavin::LINEAR, yavin::CLAMP_TO_EDGE, "color_scale.png"},
        m_noise_tex{yavin::LINEAR, yavin::REPEAT, render_resolution(0),
                    render_resolution(1)},
        m_fbotex{render_resolution(0), render_resolution(1)},
        m_seedcurvetex{render_resolution(0), render_resolution(1)},
        m_fbo{m_fbotex},
        m_rand_eng{rand_eng},
        m_domain{domain},

        m_front_v_t_t0{m_render_resolution(0), m_render_resolution(1)},
        m_front_curvature{m_render_resolution(0), m_render_resolution(1)},
        m_front_renderindex_layer{m_render_resolution(0), m_render_resolution(1)},
        m_front_depth{m_render_resolution(0), m_render_resolution(1)},
        m_front_fbo{m_front_v_t_t0, m_front_curvature, m_front_renderindex_layer,
                  m_front_depth},

        m_back_v_t_t0{m_render_resolution(0), m_render_resolution(1)},
        m_back_curvature{m_render_resolution(0), m_render_resolution(1)},
        m_back_renderindex_layer{m_render_resolution(0), m_render_resolution(1)},
        m_back_depth{m_render_resolution(0), m_render_resolution(1)},
        m_back_fbo{m_back_v_t_t0, m_back_curvature, m_back_renderindex_layer,
                 m_back_depth},
        m_lic_tex{m_render_resolution(0), m_render_resolution(1)},
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
        m_weight_buffer(m_render_resolution(0) * m_render_resolution(1), 0),
        m_num_overall_covered_pixels{0},
        m_num_newly_covered_pixels{0} {
    yavin::disable_multisampling();

    m_seedcurve_shader.set_projection(m_cam.projection_matrix());
    m_ssf_rasterization_shader.set_projection(m_cam.projection_matrix());
    m_ssf_rasterization_shader.set_width(m_render_resolution(0));
    const auto noise_data = random_uniform_vector<float>(
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
    m_weight_buffer.bind(4);

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
    m_weight_dual_pathsurface_shader.set_size(m_render_resolution(0) *
                                              m_render_resolution(1));
    // m_working_rasterization.set_usage(yavin::DYNAMIC_COPY);
    // m_working_list_size.set_usage(yavin::DYNAMIC_COPY);
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
    m_working_list_size.upload_data(0);
    yavin::shaderstorage_barrier();
    m_front_fbo.bind();
    yavin::clear_color_depth_buffer();
    yavin::depth_func_less();
    m_ssf_rasterization_shader.set_count(true);
    gpu_mesh.draw();

    m_back_fbo.bind();
    yavin::clear_color_depth_buffer();
    m_back_depth.clear(1e5);
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
  auto pathsurface(const grid_edge<real_t, 3>& edge, real_t btau, real_t ftau,
                   size_t seed_res, real_t stepsize) const {
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
      // if (seedcurve.vertex_at(0)(0) != seedcurve.vertex_at(1)(0) &&
      //    seedcurve.vertex_at(0)(1) != seedcurve.vertex_at(1)(1)) {
      simple_tri_mesh<real_t, 2> mesh =
          surf.discretize(seed_res, stepsize, btau, ftau);
      auto& uvprop   = mesh.template add_vertex_property<vec2>("uv");
      auto& vprop    = mesh.template add_vertex_property<vec2>("v");
      auto& curvprop = mesh.template add_vertex_property<real_t>("curvature");

      for (auto vertex : mesh.vertices()) {
        const auto& uv             = uvprop[vertex];
        const auto& integral_curve = surf.streamline_at(uv(0), 0, 0);
        curvprop[vertex]           = integral_curve.curvature(uv(1));
        if (std::isnan(curvprop[vertex])) {
          std::cerr << "got nan!\n";
          std::cerr << "t0     = " << u0t0 << '\n';
          std::cerr << "pos     = " << integral_curve(uv(1)) << '\n';
          std::cerr << "v       = " << m_v(integral_curve(uv(1)), uv(1)) << '\n';
          std::cerr << "tau     = " << uv(1) << '\n';
          std::cerr << "tangent = " << integral_curve.tangent(uv(1)) << '\n';
        }
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
  auto weight(GLuint layer1) -> float {
    m_weight_buffer.upload_data(0.0f);
    m_num_overall_covered_pixels[0] = 0;
    m_num_newly_covered_pixels[0]   = 0;
    m_weight_dual_pathsurface_shader.set_layer(layer1);
    m_weight_dual_pathsurface_shader.bind();
    yavin::gl::dispatch_compute(
        m_render_resolution(0) * m_render_resolution(1) / 1024.0 + 1, 1, 1);

    const auto weight_data = m_weight_buffer.download_data();
    if (m_num_newly_covered_pixels[0].download() >
        m_render_resolution(0) * m_render_resolution(1) * 0.0001) {
      return std::reduce(std::execution::par, begin(weight_data),
                         end(weight_data), 0.0f) /
             m_num_newly_covered_pixels[0].download();
    }
    return -std::numeric_limits<float>::max();
    // return gpu::reduce(m_weight_buffer, 16, 16);
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
  void result_to_lic_tex(const grid<real_t, 3>& domain) {
    const size_t num_samples = 20;
    const real_t stepsize =
        (domain.dimension(0).back() - domain.dimension(0).front()) /
        (m_render_resolution(0) * 2);
    result_to_v_tex();

    m_color_lic_tex.bind_image_texture(7);
    m_lic_tex.clear(1, 1, 1, 0);
    m_color_lic_tex.clear(1, 1, 1, 0);
    m_lic_shader.set_domain_min(domain.front(0), domain.front(1));
    m_lic_shader.set_domain_max(domain.back(0), domain.back(1));
    m_lic_shader.set_min_t(domain.front(2));
    m_lic_shader.set_max_t(domain.back(2));
    m_lic_shader.set_num_samples(num_samples);
    m_lic_shader.set_stepsize(stepsize);
    m_lic_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                          m_render_resolution(1) / 32.0 + 1);
    m_v_tex.bind_image_texture(7);
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
    m_v_tex.bind_image_texture(7);
    return curvature_tex;
  }
  //----------------------------------------------------------------------------
  /// rast1 gets written in rast0. rast0 must have additional space to be able
  /// to hold rast1.
  void combine() {
    m_combine_rasterizations_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                             m_render_resolution(1) / 32.0 + 1);
    yavin::shaderstorage_barrier();
  }
  //----------------------------------------------------------------------------
  auto integrate(
      const std::string& dataset_name,
      const std::set<std::pair<size_t, grid_edge_iterator<real_t, 3>>>&
                             unused_edges,
      const grid<real_t, 3>& domain, const real_t btau, const real_t ftau,
      const size_t seed_res, const real_t stepsize) {
    if (!std::filesystem::exists("pathsurfaces")) {
      std::filesystem::create_directory("pathsurfaces");
    }

    const auto pathsurface_dir = +"pathsurfaces/" + dataset_name + "/";
    if (!std::filesystem::exists(pathsurface_dir)) {
      std::filesystem::create_directory(pathsurface_dir);
    }

    std::string        filename_vtk;
    std::atomic_size_t progress_counter = 0;
    std::thread        t{[&] {
      float     progress  = 0.0;
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
                  << '\r';
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
      }
      for (int i = 0; i < bar_width; ++i) { std::cerr << "\u2588"; }
      std::cerr << "done!                  \n";
    }};

    for (const auto& [edge_idx, unused_edge_it] : unused_edges) {

      filename_vtk = pathsurface_dir;
      for (size_t i = 0; i < 3; ++i) {
        filename_vtk += std::to_string(domain.size(i)) + "_";
      }
      const auto min_t0 = domain.dimension(2).front();
      const auto max_t0 = domain.dimension(2).back();
      filename_vtk += std::to_string(min_t0) + "_" + std::to_string(max_t0) +
                      "_" + std::to_string(btau) + "_" + std::to_string(ftau) +
                      "_" + std::to_string(seed_res) + "_" +
                      std::to_string(stepsize) + "_" +
                      std::to_string(edge_idx) + ".vtk";
      if (!std::filesystem::exists(filename_vtk)) {
        simple_tri_mesh<real_t, 2> psf =
            pathsurface(*unused_edge_it, btau, ftau, seed_res, stepsize).first;
        psf.write_vtk(filename_vtk);
      }
      progress_counter++;
      }
      t.join();
    return pathsurface_dir;
  }
  //----------------------------------------------------------------------------
  auto setup_working_dir(const grid<real_t, 3>& domain, const real_t btau,
                         const real_t ftau, const size_t seed_res,
                         const real_t stepsize, const double neighbor_weight,
                         const float penalty) {
    using namespace std::filesystem;
    auto working_dir = std::string{settings<V>::name} + "_domain_res_";
    for (size_t i = 0; i < 3; ++i) {
      working_dir += std::to_string(domain.size(i)) + "_";
    }

    working_dir += "btau_" + std::to_string(btau) +
                   "_ftau_" + std::to_string(ftau) + "_seedres_" +
                   std::to_string(seed_res) + "_stepsize_" +
                   std::to_string(stepsize) + "_neighborweight_" +
                   std::to_string(neighbor_weight) + "_penalty_" +
                   std::to_string(penalty) + "/";
    if (!exists(working_dir)) { create_directory(working_dir); }
    std::cerr << "result will be located in " << working_dir << '\n';

    for (const auto& entry : directory_iterator(working_dir)) { remove(entry); }
    return working_dir;
  }
  void render_seedcurves(const grid<real_t, 3>&              domain,
                         const std::vector<line<real_t, 3>>& seedcurves,
                         GLfloat min_t, GLfloat max_t) {
    yavin::disable_depth_test();
    yavin::framebuffer fbo{m_seedcurvetex};
    fbo.bind();
    yavin::gl::clear_color(255, 255, 255, 0);
    yavin::clear_color_buffer();
    m_seedcurve_shader.bind();
    std::vector<line<real_t, 3>> domain_edges;

    for (auto x : domain.dimension(0)) {
      domain_edges.push_back(
          line<real_t, 3>{vec<real_t, 3>{x, domain.dimension(1).front(), 0},
                          vec<real_t, 3>{x, domain.dimension(1).back(), 0}});
    }
    for (auto y : domain.dimension(1)) {
      domain_edges.push_back(
          line<real_t, 3>{vec<real_t, 3>{domain.dimension(0).front(), y, 0},
                          vec<real_t, 3>{domain.dimension(0).back(), y, 0}});
    }

    auto domain_edges_gpu = line_renderer(domain_edges);
    m_seedcurve_shader.use_color_scale(false);
    m_seedcurve_shader.set_color(0.8f, 0.8f, 0.8f);
    domain_edges_gpu.draw();

    auto ls = line_renderer(seedcurves);
    m_seedcurve_shader.set_min_t(min_t);
    m_seedcurve_shader.set_max_t(max_t);
    m_seedcurve_shader.use_color_scale(true);
    yavin::gl::line_width(3);
    ls.draw();
    yavin::gl::line_width(1);
    yavin::enable_depth_test();
  }
  //----------------------------------------------------------------------------
  auto greedy_set_cover(const grid<real_t, 3>& domain, const real_t btau,
                        const real_t ftau, const size_t seed_res,
                        const real_t stepsize, real_t desired_coverage,
                        const double neighbor_weight, const float penalty)
      -> std::string {
    using namespace std::filesystem;
    auto   best_weight     = -std::numeric_limits<real_t>::max();
    size_t render_index    = 0;
    size_t layer           = 0;
    std::set<std::pair<size_t, grid_edge_iterator<real_t, 3>>>    unused_edges;
    std::vector<std::pair<size_t, grid_edge_iterator<real_t, 3>>> used_edges;
    auto               best_edge    = end(unused_edges);
    size_t             iteration    = 0;
    size_t             edge_counter = 0;
    bool               stop_thread  = false;
    double             coverage     = 0;
    const auto         min_t0       = domain.front(2);
    const auto         max_t0       = domain.back(2);
    std::vector<float> weights;
    std::vector<float> coverages;

    size_t best_num_usages0, best_num_usages1;
    bool   best_correct_usage0, best_correct_usage1;
    bool   best_crossed;

    const size_t num_pixels = m_render_resolution(0) * m_render_resolution(1);
    size_t num_pixels_in_domain = 0;
    for (size_t y = 0; y < m_render_resolution(1); ++y) {
      for (size_t x = 0; x < m_render_resolution(0); ++x) {
        const real_t         xt = x / (m_render_resolution(0) - 1);
        const real_t         yt = y / (m_render_resolution(1) - 1);
        const vec<real_t, 2> pos{
            (1 - xt) * domain.front(0) + xt * domain.back(0),
            (1 - yt) * domain.front(1) + yt * domain.back(1)};
        if (m_v.in_domain(pos, domain.front(2))) { ++num_pixels_in_domain; }
      }
    }
    desired_coverage *= real_t(num_pixels_in_domain) / real_t(num_pixels);
    std::cerr << "number of pixels in domain: "
              << real_t(num_pixels_in_domain) / real_t(num_pixels) * 100
              << "%\n";

    auto working_dir = setup_working_dir(domain, btau, ftau, seed_res,
                                         stepsize, neighbor_weight, penalty);
    m_weight_dual_pathsurface_shader.set_penalty(penalty);

    // set all edges as unused
    std::cerr << "set all edges unused\n";
    size_t ec = 0;
    for (auto edge_it = domain.edges().begin(); edge_it != domain.edges().end();
         ++edge_it) {
      const auto [v0, v1] = *edge_it;
      if (v0[2] == v1[2]) { unused_edges.insert(std::pair{ec++, edge_it}); }
    }

    const auto  integration_measure  = measure([&] {
      return integrate(std::string{settings<V>::name}, unused_edges, domain,
                       btau, ftau, seed_res, stepsize);
    });
    const auto& integration_duration = integration_measure.first;
    const auto& pathsurface_dir      = integration_measure.second;

    // monitoring
    std::thread t{[&] {
      double prog0 = 0.0;
      std::cerr << "cur it          coverage    \n";
      while (!stop_thread) {
        const int bar_width = 15;
        prog0               = double(edge_counter) / (unused_edges.size());
        int pos0            = bar_width * prog0;
        int pos2            = bar_width * coverage;
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

    std::vector<line<real_t, 3>> seedcurves;
    auto                         duration = measure([&] {
      // loop
      do {
        edge_counter    = 0;
        best_weight = -std::numeric_limits<real_t>::max();
        best_edge   = end(unused_edges);

        for (auto unused_edge_pair_it = begin(unused_edges);
             unused_edge_pair_it != end(unused_edges); ++unused_edge_pair_it) {
          const auto& [edge_idx, unused_edge_it] = *unused_edge_pair_it;
          const auto  unused_edge                = *unused_edge_it;
          std::string filepath_vtk               = pathsurface_dir;
          for (size_t i = 0; i < 3; ++i) {
            filepath_vtk += std::to_string(domain.size(i)) + "_";
          }
          filepath_vtk +=
              std::to_string(min_t0) + "_" + std::to_string(max_t0) + "_" +
              std::to_string(btau) + "_" + std::to_string(ftau) + "_" +
              std::to_string(seed_res) + "_" + std::to_string(stepsize) + "_" +
              std::to_string(edge_idx) + ".vtk";
          simple_tri_mesh<real_t, 2> mesh{filepath_vtk};
          if (mesh.num_faces() > 0) {
            rasterize(gpu_pathsurface(mesh, unused_edge.first.position()(2),
                                      unused_edge.second.position()(2)),
                      render_index, layer);
            auto new_weight = weight(layer);
            if (m_num_newly_covered_pixels[0] > 0) {
              // check if mesh's seedcurve neighbors another edge
              size_t num_usages0 = 0, num_usages1 = 0;
              bool   correct_usage0 = false, correct_usage1 = false;
              bool   crossed = false;
              for (const auto& [used_edge_idx, used_edge_it] : used_edges) {
                const auto [uv0, uv1] = *used_edge_it;
                const auto used_x0    = uv0.to_array();
                const auto used_x1    = uv1.to_array();
                const auto new_x0     = unused_edge.first.to_array();
                const auto new_x1     = unused_edge.second.to_array();
                // check if crossed
                if (used_x0[0] != used_x1[0] && used_x0[1] != used_x1[1] &&
                    new_x0[0] != new_x1[0] && new_x0[1] != new_x1[1] &&
                    min(used_x0[0], used_x1[0]) == min(new_x0[0], new_x1[0]) &&
                    max(used_x0[0], used_x1[0]) == max(new_x0[0], new_x1[0]) &&
                    min(used_x0[1], used_x1[1]) == min(new_x0[1], new_x1[1]) &&
                    max(used_x0[1], used_x1[1]) == max(new_x0[1], new_x1[1])) {
                  crossed = true;
                  break;
                }

                if ((used_x0[0] == new_x0[0] && used_x0[1] == new_x0[1]) ||
                    (used_x1[0] == new_x0[0] && used_x1[1] == new_x0[1])) {
                  ++num_usages0;
                  if (used_x0[2] == new_x0[2] || used_x1[2] == new_x0[2]) {
                    correct_usage0 = true;
                  }
                }
                if ((used_x0[0] == new_x1[0] && used_x0[1] == new_x1[1]) ||
                    (used_x1[0] == new_x1[0] && used_x1[1] == new_x1[1])) {
                  ++num_usages1;
                  if (used_x0[2] == new_x1[2] || used_x1[2] == new_x1[2]) {
                    correct_usage1 = true;
                  }
                }
              }

              if (crossed || num_usages0 > 1 || num_usages1 > 1) {
                new_weight /= neighbor_weight;
              } else if ((correct_usage0 && num_usages0 == 1) ||
                         (correct_usage1 && num_usages1 == 1)) {
                new_weight *= neighbor_weight;
              }
              if (new_weight > best_weight) {
                best_weight = new_weight;
                best_edge   = unused_edge_pair_it;
                best_num_usages0 = num_usages0;
                best_num_usages1 = num_usages1;
                best_correct_usage0 = correct_usage0;
                best_correct_usage1 = correct_usage1;
                best_crossed = crossed;
              }
            }
          }
          ++edge_counter;
        }
        if (best_edge != end(unused_edges)) {
          if (best_crossed) {
            std::cerr << "crossed! " << *best_edge->second << "\n";
          } else {
            if (best_correct_usage0 && best_num_usages0 == 1) {
              std::cerr << "great! " << *best_edge->second << "\n";
            } else if (best_correct_usage1 && best_num_usages1 == 1) {
              std::cerr << "great! " << *best_edge->second << "\n";
            } else if (best_num_usages0 > 1 || best_num_usages1 > 1) {
              std::cerr << "multi! " << *best_edge->second << "\n";
            }
          }
          std::string filepath_vtk = pathsurface_dir;
          for (size_t i = 0; i < 3; ++i) {
            filepath_vtk += std::to_string(domain.size(i)) + "_";
          }
          filepath_vtk +=
              std::to_string(min_t0) + "_" + std::to_string(max_t0) + "_" +
              std::to_string(btau) + "_" + std::to_string(ftau) + "_" +
              std::to_string(seed_res) + "_" + std::to_string(stepsize) + "_" +
              std::to_string(best_edge->first) + ".vtk";
          simple_tri_mesh<real_t, 2> mesh{filepath_vtk};
          auto [v0, v1] = *best_edge->second;
          rasterize(gpu_pathsurface(mesh, v0.position()(2), v1.position()(2)),
                    render_index, layer);
          combine();
          used_edges.push_back(*best_edge);

          ++render_index;
          coverage =
              static_cast<double>(m_num_overall_covered_pixels[0].download()) /
              (m_render_resolution(0) * m_render_resolution(1));
          weights.push_back(best_weight /
                            (m_render_resolution(0) * m_render_resolution(1)));
          coverages.push_back(coverage);

          std::string it_str = std::to_string(iteration);
          while (it_str.size() < 4) { it_str = '0' + it_str; }
          result_to_lic_tex(domain);
          m_lic_tex.write_png(working_dir + "lic_" + it_str + ".png");
          m_color_lic_tex.write_png(working_dir + "lic_color_" + it_str + ".png");
          const std::string mesh3dpath =
              working_dir + "geometry_" + it_str + ".vtk";
          simple_tri_mesh<real_t, 3> mesh3d;
          auto&                      uv2d_prop =
              mesh.template vertex_property<vec<real_t, 2>>("uv");
          auto& uv3d_prop =
              mesh3d.template add_vertex_property<vec<real_t, 2>>("uv");
          auto& curv2d_prop =
              mesh.template vertex_property<real_t>("curvature");
          auto& curv3d_prop =
              mesh3d.template add_vertex_property<real_t>("curvature");
          auto& v2d_prop = mesh.template vertex_property<vec<real_t, 2>>("v");
          auto& v3d_prop =
              mesh3d.template add_vertex_property<vec<real_t, 2>>("v");

          for (const auto v : mesh.vertices()) {
            const auto& x = mesh[v];
            mesh3d.insert_vertex(x(0), x(1), uv2d_prop[v](1));
            uv3d_prop[v.i]   = uv2d_prop[v];
            curv3d_prop[v.i] = curv2d_prop[v];
            v3d_prop[v.i]    = v2d_prop[v];
          }
          for (const auto f : mesh.faces()) {
            const auto& [v0, v1, v2] = mesh[f];
            mesh3d.insert_face(v0.i, v1.i, v2.i);
          }

          mesh3d.write_vtk(mesh3dpath);
        }
        if (best_edge != end(unused_edges)) { unused_edges.erase(best_edge); }
        ++iteration;
      } while (coverage < desired_coverage && best_edge != end(unused_edges));

      std::cerr << "done!\n";
      result_to_lic_tex(domain);
      m_lic_tex.write_png(working_dir + "lic_final.png");
      m_color_lic_tex.write_png(working_dir + "lic_color_final.png");
      for (const auto& [used_edge_idx, used_edge_it]: used_edges) {
        const auto [v0, v1]  = *used_edge_it;
        seedcurves.push_back(line<real_t, 3>{v0.position(), v1.position()});
      }
    });

    stop_thread = true;
    t.join();
    std::cerr << '\n';
    std::cerr << "seedcurves!\n";
    write_vtk(seedcurves, working_dir + "seedcurves.vtk");
    render_seedcurves(domain, seedcurves, domain.dimension(2).front(), domain.dimension(2).back());
    m_seedcurvetex.write_png(working_dir + "seedcurves.png");

    std::string cmd = "#/bin/bash \n";
    cmd += "cd " + working_dir + '\n';
    cmd +=
        "ffmpeg -y -r 3 -start_number 0 -i lic_%04d.png -c:v libx264 -vf "
        "fps=25 "
        "-pix_fmt yuv420p lic.mp4\n";
    cmd +=
        "ffmpeg -y -r 3 -start_number 0 -i lic_color_%04d.png -c:v libx264 -vf "
        "fps=25 "
        "-pix_fmt yuv420p lic_color.mp4\n";
    system(cmd.c_str());

    const std::string   reportfilepath = working_dir + "index.html";
    html::doc           reportfile;
    std::vector<size_t> labels(used_edges.size());
    boost::iota(labels, 0);
    reportfile.add(html::chart{weights, "weight", "#3e95cd", labels});
    reportfile.add(html::chart{coverages, "coverage", "#FF0000", labels});

    reportfile.add(html::heading{"Final LIC"}, html::image{"lic_final.png"});
    reportfile.add(html::heading{"Final LIC color coded"},
                   html::image{"lic_color_final.png"});
    reportfile.add(html::video{"lic.mp4"});
    reportfile.add(html::video{"lic_color.mp4"});

    html::slider lics;
    for (size_t i = 0; i < used_edges.size(); ++i) {
      std::string itstr = std::to_string(i);
      while (itstr.size() < 4) { itstr = '0' + itstr; }
      lics.add(html::vbox{
          html::table{std::vector{"iteration#", "weight", "coverage"},
                      std::vector{std::to_string(i), std::to_string(weights[i]),
                                  std::to_string(coverages[i])}},
          html::image{"lic_" + itstr + ".png"},
          html::image{"lic_color_" + itstr + ".png"}});
    }
    reportfile.add(lics);
    reportfile.add(html::heading{"seedcurves color coded"},
                   html::image{"seedcurves.png"});
    reportfile.add(html::table{
        std::vector{"btau", "ftau", "seed res", "stepsize", "coverage",
                    "neighbor weight", "penalty", "num iterations"},
        std::vector{std::to_string(btau), std::to_string(ftau),
                    std::to_string(seed_res), std::to_string(stepsize),
                    std::to_string(coverage), std::to_string(neighbor_weight),
                    std::to_string(penalty),
                    std::to_string(used_edges.size())}});
    const auto [min, s, ms] =
        break_down_durations<std::chrono::minutes, std::chrono::seconds,
                             std::chrono::milliseconds>(duration);
    const auto [min2, s2, ms2] =
        break_down_durations<std::chrono::minutes, std::chrono::seconds,
                             std::chrono::milliseconds>(integration_duration);
    std::stringstream tstr;
    tstr << min.count() << ":" << s.count() << ":" << ms.count() << " min";

    std::stringstream tstr2;
    tstr2 << min2.count() << ":" << s2.count() << ":" << ms2.count() << " min";

    reportfile.add(html::heading{"Steadification Timing"},
                   html::text{tstr.str()}, html::heading{"Integration Timing"},
                   html::text{tstr2.str()});

    reportfile.write(reportfilepath);
    return working_dir;
  }
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
