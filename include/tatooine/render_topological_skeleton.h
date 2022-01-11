#ifndef TATOOINE_RENDER_TOPOLOGICAL_SKELETON
#define TATOOINE_RENDER_TOPOLOGICAL_SKELETON
//==============================================================================
#include <tatooine/diff.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/topological_skeleton.h>
#include <tatooine/grid_sampler.h>
#include <tatooine/sampled_field.h>
#include <tatooine/gl/context.h>
#include <tatooine/gl/framebuffer.h>
#include <tatooine/gl/glwrapper.h>
#include <tatooine/gl/indexbuffer.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/orthographiccamera.h>
#include <tatooine/gl/vertexarray.h>
#include <tatooine/gl/vertexbuffer.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
/// you need a gl::context for this
template <typename Real, typename Integrator,
          template <typename> typename Interpolator>
auto render_topological_skeleton(
    const sampled_field<
        grid_sampler<Real, 2, vec<Real, 2>, interpolation::linear,
                     interpolation::linear>,
        Real, 2, 2>& v,
    const integration::integrator<Real, 2, Interpolator, Integrator>&
                          integrator,
    const vec<size_t, 2>& resolution,
    const Real tau = 100, const Real eps = 1e-7) {
  using namespace gl;
  struct point_shader : shader {
    point_shader() {
      add_stage<vertexshader>(
          "#version 450\n"
          "layout(location = 0) in vec2 pos;\n"
          "layout(location = 1) in vec3 col;\n"
          "out vec2 geom_pos;\n"
          "out vec3 geom_col;\n"
          "void main() {\n"
          "  geom_pos = pos;\n"
          "  geom_col = col;\n"
          "}",
          shaderstageparser::SOURCE);
      add_stage<geometryshader>(
          "#version 450\n"
          "#define pi 3.1415926535897932384626433832795\n"
          "layout (points) in;\n"
          "layout (triangle_strip, max_vertices = 100) out;\n"
          "\n"
          "uniform mat4 projection;\n"
          "uniform vec2 domain_min;\n"
          "uniform vec2 domain_max;\n"
          "uniform float radius;\n"
          "\n"
          "in vec2[] geom_pos;\n"
          "in vec3[] geom_col;\n"
          "\n"
          "out vec3  frag_col;\n"
          "out float frag_param;\n"
          "\n"
          "void main() {\n"
          "  uint  n = 20;\n"
          "  frag_col = geom_col[0];\n"
          "  for (uint i = 0; i < n - 1; ++i) {\n"
          "    gl_Position = projection * vec4(geom_pos[0], 0, 1);\n"
          "    frag_param = 0;\n"
          "    EmitVertex();\n"
          "    gl_Position = projection * vec4(geom_pos[0] +\n"
          "                    vec2(cos(float(i) / float(n-1) * 2*pi),\n"
          "                         sin(float(i) / float(n-1) * 2*pi)) * radius,\n"
          "                     0, 1);\n"
          "    frag_param = 1;\n"
          "    EmitVertex();\n"
          "    gl_Position = projection * vec4(geom_pos[0] +\n"
          "                         vec2(cos(float(i+1) / float(n-1) * 2*pi),\n"
          "                              sin(float(i+1) / float(n-1) * 2*pi)) * radius,\n"
          "                       0, 1);\n"
          "    frag_param = 1;\n"
          "    EmitVertex();\n"
          "    EndPrimitive();\n"
          "  }\n"
          "}",
          shaderstageparser::SOURCE);
      add_stage<fragmentshader>(
          "#version 450\n"
          "in vec3 frag_col;\n"
          "in float frag_param;\n"
          "layout(location = 0) out vec4 frag;\n"
          "void main() {\n"
          "  if (frag_param  > 0.8) {\n"
          "    frag = vec4(vec3(0), 1);\n"
          "  } else  {\n"
          "    frag = vec4(frag_col, 1);\n"
          "  }\n"
          "}",
          shaderstageparser::SOURCE);
      create();
    }
    void set_projection(const glm::mat4x4& p) { set_uniform("projection", p); }
    void set_radius(const GLfloat radius) { set_uniform("radius", radius); }
  };
  //==============================================================================
  struct line_shader : shader {
    line_shader() {
      add_stage<vertexshader>(
          "#version 450\n"
          "uniform mat4 projection;\n"
          "layout(location = 0) in vec2 pos;\n"
          "void main() {\n"
          "  gl_Position = projection * vec4(pos, 0, 1);\n"
          "}\n",
          shaderstageparser::SOURCE);
      add_stage<fragmentshader>(
          "#version 450\n"
          "layout(location = 0) out vec4 frag;\n"
          "void main() {\n"
          "  frag = vec4(0,0,0,1);\n"
          "}\n",
          shaderstageparser::SOURCE);
      create();
    }
    void set_projection(const glm::mat4x4& p) { set_uniform("projection", p); }
  };
  using namespace interpolation;
  using gpu_point_data_t = indexeddata<vec2f, vec3f>;
  using gpu_line_data_t  = indexeddata<vec2f>;
  using sampler_t = grid_sampler<double, 2, vec<double, 2>, linear, linear>;

  gpu_point_data_t::vbo_data_vec vbo_point_data;
  gpu_point_data_t::ibo_data_vec ibo_point_data;
  gpu_line_data_t::vbo_data_vec vbo_line_data;
  gpu_line_data_t::ibo_data_vec ibo_line_data;

  auto skel = compute_topological_skeleton(v, integrator);

  // upload critical points
  for (const auto& x : skel.saddles) {
    vbo_point_data.push_back({{float(x(0)), float(x(1))}, {1.0f, 1.0f, .0f}});
    ibo_point_data.push_back(ibo_point_data.size());
  }
  for (const auto& x : skel.sinks) {
    vbo_point_data.push_back({{float(x(0)), float(x(1))}, {1.0f, .0f, .0f}});
    ibo_point_data.push_back(ibo_point_data.size());
  }
  for (const auto& x : skel.sources) {
    vbo_point_data.push_back({{float(x(0)), float(x(1))}, {.0f, .0f, 1.0f}});
    ibo_point_data.push_back(ibo_point_data.size());
  }
  for (const auto& x : skel.attracting_foci) {
    vbo_point_data.push_back({{float(x(0)), float(x(1))}, {1.0f, .25f, .25f}});
    ibo_point_data.push_back(ibo_point_data.size());
  }
  for (const auto& x : skel.repelling_foci) {
    vbo_point_data.push_back({{float(x(0)), float(x(1))}, {.25f, .25f, 1.0f}});
    ibo_point_data.push_back(ibo_point_data.size());
  }
  for (const auto& x : skel.centers) {
    vbo_point_data.push_back({{float(x(0)), float(x(1))}, {.0f, 0.75f, 0.0f}});
    ibo_point_data.push_back(ibo_point_data.size());
  }
  for (const auto& x : skel.boundary_switch_points) {
    vbo_point_data.push_back({{float(x(0)), float(x(1))}, {1.0f, 1.0f, 1.0f}});
    ibo_point_data.push_back(ibo_point_data.size());
  }
  gpu_point_data_t gpu_point_data{vbo_point_data, ibo_point_data};
  gpu_point_data.setup_vao();

  // upload separatrices
  size_t                        line_cnt = 0;
  for (const auto& separatrix : skel.separatrices) {
    for (size_t i = 0; i < separatrix.vertices().size(); ++i) {
      vbo_line_data.push_back({static_cast<float>(separatrix.vertex_at(i)(0)),
                               static_cast<float>(separatrix.vertex_at(i)(1))});
    }
    for (size_t i = 0; i < separatrix.vertices().size() - 1; ++i) {
      ibo_line_data.push_back(line_cnt);
      ibo_line_data.push_back(++line_cnt);
    }
    ++line_cnt;
  }
  gpu_line_data_t gpu_line_data{vbo_line_data, ibo_line_data};
  gpu_line_data.setup_vao();

  // create lic texture
  const Real pixel_size =
      std::min((v.sampler().back(0) - v.sampler().front(0)) / resolution(0),
               (v.sampler().back(1) - v.sampler().front(1)) / resolution(1));
  const Real lic_stepsize = pixel_size / 4;
  const size_t lic_num_samples = 100;
  auto image = gpu::lic(v.sampler(), resolution, lic_num_samples, lic_stepsize,
                        {256, 256});
  orthographiccamera cam(v.sampler().front(0), v.sampler().back(0),
                         v.sampler().front(1), v.sampler().back(1), -10, 10, 0,
                         0, resolution(0), resolution(1));

  // render separatrices and critical points over lic image
  framebuffer fbo{image};
  fbo.bind();
  gl::viewport(cam.viewport());
  {
    // render lines
    line_shader shader;
    shader.bind();
    shader.set_projection(cam.projection_matrix());
    gl::line_width(3);
    gpu_line_data.draw_lines();
  }
  {
    // render points
    point_shader shader;
    shader.bind();
    shader.set_projection(cam.projection_matrix());
    shader.set_radius(pixel_size*10);
    gpu_point_data.draw_points();
  }
  fbo.unbind();

  return image;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
