#ifndef TATOOINE_STEADIFICATION_STEADIFICATION_H
#define TATOOINE_STEADIFICATION_STEADIFICATION_H

#include <tatooine/chrono.h>
#include <tatooine/for_loop.h>

#include <cstdlib>
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
#include "integrate.h"
#include "pathsurface.h"

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
  using pathsurface_gpu_t    = streamsurface_renderer;
  using vec2                 = vec<real_t, 2>;
  using vec3                 = vec<real_t, 3>;
  using ivec2                = vec<size_t, 2>;
  using ivec3                = vec<size_t, 3>;
  using grid_t               = grid<real_t, 2>;
  using grid_edge_iterator_t = typename grid_t::edge_iterator;

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
  ll_to_v_shader                     m_result_to_v_tex_shader;
  weight_shader                      m_weight_shader;
  RandEng&                           m_rand_eng;
  boundingbox<real_t, 2>             m_domain;
  combine_rasterizations_shader      m_combine_rasterizations_shader;
  seedcurve_shader                   m_seedcurve_shader;
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
  yavin::tex2rgba32f m_curvature_lic_tex;
  yavin::tex2rgba32f m_color_lic_tex;
  yavin::tex2rgba32f m_v_tex;

  yavin::shaderstoragebuffer<node>    m_result_rasterization;
  yavin::shaderstoragebuffer<node>    m_working_rasterization;
  yavin::shaderstoragebuffer<GLuint>  m_result_list_size;
  yavin::shaderstoragebuffer<GLuint>  m_working_list_size;
  yavin::shaderstoragebuffer<GLfloat> m_weight_buffer;

  size_t m_num_totally_covered_pixels = 0;
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
        m_weight_buffer(m_render_resolution(0) * m_render_resolution(1), 0),
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
    m_num_newly_covered_pixels.bind(0);
    m_v_tex.bind(0);
    m_noise_tex.bind(1);
    m_color_scale.bind(2);
    m_ssf_rasterization_shader.set_width(m_render_resolution(0));
    m_combine_rasterizations_shader.set_resolution(m_render_resolution(0),
                                                   m_render_resolution(1));
    m_lic_shader.set_v_tex_bind_point(0);
    m_lic_shader.set_noise_tex_bind_point(1);
    m_lic_shader.set_color_scale_bind_point(2);
    m_weight_shader.set_size(m_render_resolution(0) * m_render_resolution(1));
    yavin::gl::viewport(m_cam);
    yavin::enable_depth_test();
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
    yavin::clear_color_depth_buffer();
    yavin::depth_func_less();
    m_ssf_rasterization_shader.set_count(true);
    yavin::barrier();
    gpu_mesh.draw();
    yavin::barrier();

    m_back_fbo.bind();
    yavin::clear_color_depth_buffer();
    m_back_depth.clear(1e5);
    yavin::depth_func_greater();
    m_ssf_rasterization_shader.set_count(false);
    yavin::barrier();
    gpu_mesh.draw();
    yavin::barrier();
  }
  //----------------------------------------------------------------------------
  void to_shaderstoragebuffer() {
    yavin::barrier();
    m_tex_rasterization_to_buffer_shader.dispatch(
        m_render_resolution(0) / 32.0 + 1, m_render_resolution(1) / 32.0 + 1);
    yavin::barrier();
  }
  //----------------------------------------------------------------------------
  auto weight(GLboolean use_tau, bool normalize_weight) -> float {
    m_weight_buffer.upload_data(0.0f);
    m_num_newly_covered_pixels[0] = 0;
    m_weight_shader.bind();
    m_weight_shader.use_tau(use_tau);
    yavin::barrier();
    yavin::gl::dispatch_compute(
        m_render_resolution(0) * m_render_resolution(1) / 1024.0 + 1, 1, 1);
    yavin::barrier();

    const auto weight_data = m_weight_buffer.download_data();
    if (normalize_weight) {
    const auto num_newly_covered_pixels = m_num_newly_covered_pixels[0].download();
      if (num_newly_covered_pixels >
          m_render_resolution(0) * m_render_resolution(1) * 0.01) {
        return std::reduce(std::execution::par, begin(weight_data),
                           end(weight_data), 0.0f) /
               num_newly_covered_pixels;
      } else {
        return -std::numeric_limits<float>::max();
      }
    } else {
      return std::reduce(std::execution::par, begin(weight_data),
                         end(weight_data), 0.0f);
    }
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
    yavin::barrier();
    m_result_to_v_tex_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                      m_render_resolution(1) / 32.0 + 1);
    yavin::barrier();
  }
  //----------------------------------------------------------------------------
  /// rast1 gets written in rast0. rast0 must have additional space to be able
  /// to hold rast1.
  void combine() {
    m_combine_rasterizations_shader.dispatch(m_render_resolution(0) / 32.0 + 1,
                                             m_render_resolution(1) / 32.0 + 1);
    yavin::barrier();
  }
  //----------------------------------------------------------------------------
  auto setup_working_dir(const grid_t& domain, const real_t min_btau,
                         const real_t max_ftau, const real_t min_t,
                         const real_t max_t, const std::vector<real_t>& t0s,
                         const size_t seed_res, const real_t stepsize,
                         const double neighbor_weight, const float penalty,
                         const float max_curvature, const bool use_tau,
                         const bool normalize_weight) {
    using namespace std::filesystem;
    auto working_dir = std::string{settings<V>::name} + "/domain_res_";
    for (size_t i = 0; i < 2; ++i) {
      working_dir += std::to_string(domain.size(i)) + "_";
    }

    working_dir += "/btau_" + std::to_string(min_btau) +
                   "_ftau_" + std::to_string(max_ftau) +
                   "_min_t_" + std::to_string(min_t) +
                   "_max_t_" + std::to_string(max_t) + 
                   "_t0_";
    for (auto t0 : t0s) { working_dir += std::to_string(t0) + "_"; }
    working_dir += "seedres_" + std::to_string(seed_res) +
                   "_stepsize_" + std::to_string(stepsize) +
                   "/normalizeweight_" + (normalize_weight ? "true" : "false") +
                   "/usetau_" + (use_tau ? "true" : "false") +
                   "/neighborweight_" + std::to_string(neighbor_weight) +
                   "/penalty_" + std::to_string(penalty) +
                   "/maxcurvature_" + std::to_string(max_curvature) +
                    "/";
    if (!exists(working_dir)) { create_directories(working_dir); }
    std::cerr << "result will be located in " << working_dir << '\n';

    for (const auto& entry : directory_iterator(working_dir)) { remove(entry); }
    return working_dir;
  }
  void render_seedcurves(const grid_t&              domain,
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
    m_seedcurve_shader.set_color(0.8f, 0.8f, 0.8f, 1.0f);
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
  auto greedy_set_cover(const grid_t& domain, const real_t min_t,
                        const real_t max_t, const std::vector<real_t>& t0s,
                        const real_t min_btau, const real_t max_ftau,
                        const size_t seed_res, const real_t stepsize,
                        real_t desired_coverage, const double neighbor_weight,
                        const float penalty, const float max_curvature,
                        const bool use_tau, const bool normalize_weight)
      -> std::string {
    using namespace std::string_literals;
    using namespace std::filesystem;
    const real_t t_center = (min_t + max_t) * 0.5;
    m_weight_shader.set_t_center(t_center);
    auto   best_weight  = -std::numeric_limits<float>::max();
    size_t render_index = 0;
    size_t layer        = 0;
    std::set<std::tuple<size_t, grid_edge_iterator_t, real_t>>    unused_edges;
    std::vector<std::tuple<size_t, grid_edge_iterator_t, real_t>> used_edges;
    auto               best_edge_tuple_it = end(unused_edges);
    size_t             iteration          = 0;
    size_t             edge_counter       = 0;
    bool               stop_thread        = false;
    double             coverage           = 0;
    std::vector<float> weights;
    std::vector<float> coverages;

    size_t best_num_usages0, best_num_usages1;
    bool   best_crossed;
    size_t best_num_newly_covered_pixels = 0;

    const size_t num_pixels = m_render_resolution(0) * m_render_resolution(1);
    size_t       num_pixels_in_domain = 0;
    for (size_t y = 0; y < m_render_resolution(1); ++y) {
      for (size_t x = 0; x < m_render_resolution(0); ++x) {
        const real_t         xt = real_t(x) / real_t(m_render_resolution(0) - 1);
        const real_t         yt = real_t(y) / real_t(m_render_resolution(1) - 1);
        const vec<real_t, 2> pos{
            (1 - xt) * domain.front(0) + xt * domain.back(0),
            (1 - yt) * domain.front(1) + yt * domain.back(1)};
        if (m_v.in_domain(pos, min_t)) { ++num_pixels_in_domain; }
      }
    }
    desired_coverage *= real_t(num_pixels_in_domain) / real_t(num_pixels);
    std::cerr << "render resolution: " << m_render_resolution(0) << " x "
              << m_render_resolution(1) << "\n";
    std::cerr << "number of pixels in domain: "
              << real_t(num_pixels_in_domain) / real_t(num_pixels) * 100
              << "%\n";

    auto working_dir = setup_working_dir(
        domain, min_btau, max_ftau, min_t, max_t, t0s, seed_res, stepsize, neighbor_weight, penalty,
        max_curvature, use_tau, normalize_weight);
    m_weight_shader.set_penalty(penalty);
    m_lic_shader.set_max_curvature(max_curvature);
    m_weight_shader.set_max_curvature(max_curvature);

    // set all edges as unused
    std::cerr << "set all edges unused\n";
    for (auto t0 : t0s) {
      size_t ec = 0;
      for (auto edge_it = domain.edges().begin();
           edge_it != domain.edges().end(); ++edge_it) {
        unused_edges.insert(std::tuple{ec++, edge_it, t0});
      }
    }

    const auto  integration_measure  = measure([&] {
      return integrate(m_v, std::string{settings<V>::name}, unused_edges,
                       domain, min_t, max_t, min_btau, max_ftau, seed_res, stepsize);
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
        std::cerr << coverage << " / " << desired_coverage << '\r';
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
      }
      std::cerr << "\n";
    }};

    std::vector<line<real_t, 3>> seedcurves;
    auto                         duration = measure([&] {
      // loop
      do {
        edge_counter                  = 0;
        best_weight                   = -std::numeric_limits<float>::max();
        best_edge_tuple_it            = end(unused_edges);
        best_num_newly_covered_pixels = 0;

        for (auto unused_edge_tuple_it = begin(unused_edges);
             unused_edge_tuple_it != end(unused_edges);
             ++unused_edge_tuple_it) {
          const auto& [edge_idx, unused_edge_it, unused_t0] =
              *unused_edge_tuple_it;
          const auto [unused_v0, unused_v1]          = *unused_edge_it;
          std::string filepath_vtk                   = pathsurface_dir;
          for (size_t i = 0; i < 2; ++i) {
            filepath_vtk += std::to_string(domain.size(i)) + "_";
          }
          const auto ftau = min(max_ftau, max_t - unused_t0);
          const auto btau = max(min_btau, min_t - unused_t0);
          filepath_vtk += std::to_string(unused_t0) + "_" + std::to_string(btau) +
                          "_" + std::to_string(ftau) + "_" +
                          std::to_string(seed_res) + "_" +
                          std::to_string(stepsize) + "_" +
                          std::to_string(edge_idx) + ".vtk";
          simple_tri_mesh<real_t, 2> mesh{filepath_vtk};
          size_t                     num_usages0 = 0, num_usages1 = 0;
          bool                       crossed = false;
          real_t min_angle0 = 2 * M_PI, min_angle1 = 2 * M_PI;
          bool   same_time0 = true, same_time1 = true;

          if (mesh.num_faces() > 0) {
            rasterize(pathsurface_gpu_t{mesh, unused_t0}, render_index, layer);
            auto new_weight = weight(use_tau, normalize_weight);
            const auto num_newly_covered_pixels =
                m_num_newly_covered_pixels[0].download();
            // check if mesh's seedcurve neighbors another edge
            for (const auto& [used_edge_idx, used_edge_it, used_t0] :
                 used_edges) {
              const auto [used_v0, used_v1] = *used_edge_it;
              // check if crossed
              if (// used is diagonal
                  used_v0[0] != used_v1[0] &&
                  used_v0[1] != used_v1[1] &&
                  // unused is diagonal
                  unused_v0[0] != unused_v1[0] &&
                  unused_v0[1] != unused_v1[1] &&
                  // lowest x is same for unused and used
                  min(  used_v0[0],   used_v1[0]) ==
                  min(unused_v0[0], unused_v1[0]) &&
                  //// highest x is same for unused and used
                  //max(  used_v0[0],   used_v1[0]) ==
                  //max(unused_v0[0], unused_v1[0]) &&
                  // lowest y is same for unused and used
                  min(  used_v0[1],   used_v1[1]) ==
                  min(unused_v0[1], unused_v1[1])
                  //&&
                  //// highest y is same for unused and used
                  //max(  used_v0[1],   used_v1[1]) ==
                  //max(unused_v0[1], unused_v1[1])
                  ) {
                crossed = true;
                break;
              }

              if (used_v0[0] == unused_v0[0] &&
                  used_v0[1] == unused_v0[1]) {
                ++num_usages0;
                if (unused_t0 != used_t0) {same_time0 = false;}
                min_angle0 = min<real_t>(
                    min_angle0, angle(used_v1.position() - used_v0.position(),
                                      unused_v1.position() - unused_v0.position()));
              }
              if (used_v1[0] == unused_v0[0] &&
                  used_v1[1] == unused_v0[1]) {
                ++num_usages0;
                if (unused_t0 != used_t0) {same_time0 = false;}
                min_angle0 = min<real_t>(
                    min_angle0, angle(used_v0.position() - used_v1.position(),
                                      unused_v1.position() - unused_v0.position()));
              }
              if (used_v0[0] == unused_v1[0] &&
                  used_v0[1] == unused_v1[1]) {
                ++num_usages1;
                if (unused_t0 != used_t0) { same_time1 = false; }
                min_angle1 = min<real_t>(
                    min_angle1, angle(used_v1.position() - used_v0.position(),
                                      unused_v0.position() - unused_v1.position()));
              }
              if (used_v1[0] == unused_v1[0] &&
                  used_v1[1] == unused_v1[1]) {
                ++num_usages1;
                if (unused_t0 != used_t0) { same_time1 = false; }
                min_angle1 = min<real_t>(
                    min_angle1, angle(used_v0.position() - used_v1.position(),
                                      unused_v0.position() - unused_v1.position()));
              }
            }

            if (crossed || num_usages0 > 1 || num_usages1 > 1) {
              new_weight = -std::numeric_limits<float>::max();
            } else {
              const real_t upper = 120.0 / 180.0 * M_PI;
              const real_t lower = 60.0 / 180.0 * M_PI;
              if (num_usages0 == 1 && same_time0) {
                if (min_angle0 > upper) {
                  new_weight *= neighbor_weight;
                } else if (min_angle0 < lower) {
                  new_weight = -std::numeric_limits<float>::max();
                }
              } else if (num_usages1 == 1 && same_time1) {
                if (min_angle1 > upper) {
                  new_weight *= neighbor_weight;
                } else if (min_angle1 < lower) {
                  new_weight = -std::numeric_limits<float>::max();
                }
              }
            }
            if (new_weight > best_weight) {
              best_weight        = new_weight;
              best_edge_tuple_it = unused_edge_tuple_it;
              best_num_usages0   = num_usages0;
              best_num_usages1   = num_usages1;
              best_crossed       = crossed;
              best_num_newly_covered_pixels = num_newly_covered_pixels;
            }
          }
          ++edge_counter;
        }
        if (best_edge_tuple_it != end(unused_edges) &&
            best_weight != -std::numeric_limits<float>::max()) {
          m_num_totally_covered_pixels += best_num_newly_covered_pixels;
          coverage =
              static_cast<double>(m_num_totally_covered_pixels) /
              (m_render_resolution(0) * m_render_resolution(1));
          coverages.push_back(coverage);
          weights.push_back(best_weight /
                            (m_render_resolution(0) * m_render_resolution(1)));

          used_edges.push_back(*best_edge_tuple_it);
          unused_edges.erase(best_edge_tuple_it);
          const auto& [best_edge_idx, best_edge_it, best_t0] = used_edges.back();

          if (best_crossed) {
            std::cerr << "crossed! " << *best_edge_it << "\n";
          } else if (best_num_usages0 > 1 || best_num_usages1 > 1) {
            std::cerr << "multi! " << *best_edge_it << "\n";
          }
          std::string filepath_vtk = pathsurface_dir;
          for (size_t i = 0; i < 2; ++i) {
            filepath_vtk += std::to_string(domain.size(i)) + "_";
          }
            const auto ftau = min(max_ftau, max_t - best_t0);
            const auto btau = max(min_btau, min_t - best_t0);
          filepath_vtk += std::to_string(best_t0)  + "_" +
                          std::to_string(btau)     + "_" +
                          std::to_string(ftau)     + "_" +
                          std::to_string(seed_res) + "_" +
                          std::to_string(stepsize) + "_" +
                          std::to_string(best_edge_idx) + ".vtk";
          simple_tri_mesh<real_t, 2> mesh{filepath_vtk};
          rasterize(pathsurface_gpu_t{mesh, best_t0}, render_index, layer);
          combine();

          ++render_index;

          std::string it_str = std::to_string(iteration);
          while (it_str.size() < 4) { it_str = '0' + it_str; }
          result_to_lic_tex(domain, min_t, max_t);
          m_lic_tex.write_png(working_dir + "lic_" + it_str + ".png");
          m_color_lic_tex.write_png(working_dir + "lic_color_" + it_str +
                                    ".png");
          m_curvature_lic_tex.write_png(working_dir + "lic_curvature_" + it_str +
                                    ".png");
          const std::string mesh3dpath =
              working_dir + "geometry_" + it_str + ".vtk";
          simple_tri_mesh<real_t, 3> mesh3d;
          auto& uv2d_prop = mesh.template vertex_property<vec<real_t, 2>>("uv");
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
        ++iteration;
      } while (coverage < desired_coverage &&
               best_edge_tuple_it != end(unused_edges) &&
               best_weight != -std::numeric_limits<float>::max());

      std::cerr << "done!\n";
      result_to_lic_tex(domain, min_t, max_t);
      m_lic_tex.write_png(working_dir + "lic_final.png");
      m_color_lic_tex.write_png(working_dir + "lic_color_final.png");
      for (const auto& [used_edge_idx, used_edge_it, t0] : used_edges) {
        const auto [v0, v1] = *used_edge_it;
        const auto x0 = v0.position();
        const auto x1 = v1.position();
        seedcurves.push_back(
            line{vec{x0(0), x0(1), t0}, vec{x1(0), x1(1), t0}});
      }
    });

    stop_thread = true;
    t.join();
    std::cerr << '\n';

    // write report
    write_vtk(seedcurves, working_dir + "seedcurves.vtk");
    render_seedcurves(domain, seedcurves, min_t, max_t);
    m_seedcurvetex.write_png(working_dir + "seedcurves.png");

    std::cerr << "building report file... ";
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
          html::image{"lic_color_" + itstr + ".png"},
          html::image{"lic_curvature_" + itstr + ".png"}});
    }
    reportfile.add(lics);
    reportfile.add(html::heading{"seedcurves color coded"},
                   html::image{"seedcurves.png"});
    std::string t0string = std::to_string(t0s.front());
    for (size_t i = 1; i < size(t0s); ++i) {
      t0string += ", " + std::to_string(t0s[i]);
    }
    reportfile.add(html::table{
        std::vector{"btau", "ftau", "min_t", "max_t", "t0", "seed res",
                    "stepsize", "desired coverage", "coverage",
                    "neighbor weight", "penalty", "max_curvature", "use tau",
                    "normalize_weight", "num iterations"},

        std::vector{std::to_string(min_btau), std::to_string(max_ftau),
                    std::to_string(min_t), std::to_string(max_t), t0string,
                    std::to_string(seed_res), std::to_string(stepsize),
                    std::to_string(desired_coverage), std::to_string(coverage),
                    std::to_string(neighbor_weight), std::to_string(penalty),
                    std::to_string(max_curvature),
                    (use_tau ? "true"s : "false"s),
                    (normalize_weight ? "true"s : "false"s),
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
    std::cerr << "done!\n";

    std::string cmd = "#/bin/bash\n";
    cmd +=
        "cd " + working_dir + " &>/dev/null\n"
        "ffmpeg -y -r 3 -start_number 0 -i lic_%04d.png -c:v libx264 -vf "
        "fps=25 -pix_fmt yuv420p lic.mp4 &>/dev/null\n"
        "ffmpeg -y -r 3 -start_number 0 -i lic_color_%04d.png -c:v libx264 -vf "
        "fps=25 -pix_fmt yuv420p lic_color.mp4 &>/dev/null\n";
    std::cerr << "rendering movies... ";
    system(cmd.c_str());
    std::cerr << "done!\n";

    return working_dir;
  }
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
