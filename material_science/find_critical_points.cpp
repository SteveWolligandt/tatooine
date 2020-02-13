#include <tatooine/critical_points.h>
#include <tatooine/boundary_switch.h>
#include <tatooine/diff.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/vtk_legacy.h>
#include <yavin/context.h>
#include <yavin/framebuffer.h>
#include <yavin/glwrapper.h>
#include <yavin/indexbuffer.h>
#include <yavin/indexeddata.h>
#include <yavin/orthographiccamera.h>
#include <yavin/vertexarray.h>
#include <yavin/vertexbuffer.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
using namespace yavin;
//==============================================================================
static const std::array<std::string_view, 4> filenames{
    "texture_case1.vtk", "texture_case2.vtk", "texture_case3.vtk",
    "texture_case4.vtk"};
//==============================================================================
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
        "\n"
        "in vec2[] geom_pos;\n"
        "in vec3[] geom_col;\n"
        "\n"
        "out vec3  frag_col;\n"
        "out float frag_param;\n"
        "\n"
        "void main() {\n"
        "  float r = 0.003;\n"
        "  uint  n = 20;\n"
        "  frag_col = geom_col[0];\n"
        "  for (uint i = 0; i < n - 1; ++i) {\n"
        "    gl_Position = vec4(geom_pos[0], 0, 1) * projection;\n"
        "    frag_param = 0;\n"
        "    EmitVertex();\n"
        "    gl_Position = vec4(geom_pos[0] +\n"
        "                         vec2(cos(float(i) / float(n-1) * 2*pi),\n"
        "                              sin(float(i) / float(n-1) * 2*pi))*r,\n"
        "                       0, 1) * projection;\n"
        "    frag_param = 1;\n"
        "    EmitVertex();\n"
        "    gl_Position = vec4(geom_pos[0] +\n"
        "                         vec2(cos(float(i+1) / float(n-1) * 2*pi),\n"
        "                              sin(float(i+1) / float(n-1) * 2*pi))*r,\n"
        "                       0, 1) * projection;\n"
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
        //"  } else if (frag_param  > 0.9) {\n"
        //"    frag = vec4(frag_col, 1);\n"
        "  } else  {\n"
        "    frag = vec4(frag_col, 1);\n"
        "  }\n"
        "}",
        shaderstageparser::SOURCE);
    create();
  }
  void set_projection(const glm::mat4x4& p) { set_uniform("projection", p); }
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
//==============================================================================
void write_points(std::vector<tatooine::vec<double, 2>>& points,
                  const std::string&                     path) {
  using namespace tatooine;
  using namespace boost;
  using namespace adaptors;
  std::vector<vec<double, 3>> points3(points.size());
  copy(points | transformed([](const auto& p2) {
         return vec{p2(0), p2(1), 0};
       }),
       points3.begin());
  vtk::legacy_file_writer          writer(path, vtk::POLYDATA);
  std::vector<std::vector<size_t>> verts(1, std::vector<size_t>(points.size()));
  for (size_t i = 0; i < points.size(); ++i) { verts.back()[i] = i; }
  if (writer.is_open()) {
    writer.set_title("critical points");
    writer.write_header();
    writer.write_points(points3);
    writer.write_vertices(verts);
    writer.close();
  }
}
//------------------------------------------------------------------------------
int main() {
  yavin::context ctx;
  using namespace tatooine;
  using namespace interpolation;
  using integrator_t =
      integration::vclibs::rungekutta43<double, 2, interpolation::linear>;
  using integral_curve_t = integrator_t::integral_t;
  using gpu_point_data_t = indexeddata<yavin::vec2, yavin::vec3>;
  using gpu_line_data_t  = indexeddata<yavin::vec2>;
  using sampler_t = grid_sampler<double, 2, vec<double, 2>, linear, linear>;
  using V         = sampled_field<sampler_t, double, 2, 2>;

  for (const auto& filename : filenames) {
    V v;
    v.sampler().read_vtk_scalars(std::string{filename}, "vector_field");
    auto vn = normalize(v);
    auto J  = diff(v);
    integrator_t rk43{integration::vclibs::max_step     = 0.001,
                      integration::vclibs::initial_step = 0.001};

    const vec<size_t, 2> lic_res{1500, 1500};
    auto lic_tex = gpu::lic(v.sampler(), lic_res, 100, 0.00005, {256, 256});

    // calculate critical points
    auto cps = find_critical_points(v.sampler());

    // specify critical points
    gpu_point_data_t::vbo_data_vec vbo_point_data;
    gpu_point_data_t::ibo_data_vec ibo_point_data;
    std::vector<vec<double, 2>>    saddles;
    std::vector<vec<double, 2>>    saddle_eig_vals;
    std::vector<mat<double, 2, 2>> saddle_eig_vecs;
    std::vector<vec<double, 2>>    non_saddles;
    for (const auto& cp : cps) {
      auto j                  = J(cp);
      auto [eigvecs, eigvals] = eigenvectors(j);

      // check for swirling behavior
      if (std::abs(eigvals(0).imag()) < 1e-10 &&
          std::abs(eigvals(1).imag()) < 1e-10) {
        if ((eigvals(0).real() < 0 && eigvals(1).real() > 0) ||
            (eigvals(0).real() > 0 && eigvals(1).real() < 0)) {
          // saddle
          saddles.push_back(cp);
          saddle_eig_vals.push_back(real(eigvals));
          saddle_eig_vecs.push_back(real(eigvecs));
          vbo_point_data.push_back(
              {{float(cp(0)), float(cp(1))}, {1.0f, 1.0f, .0f}});
        } else if (eigvals(0).real() < 0 && eigvals(1).real() < 0) {
          // sink
          non_saddles.push_back(cp);
          vbo_point_data.push_back(
              {{float(cp(0)), float(cp(1))}, {1.0f, .0f, .0f}});
        } else /*if (eigvals(0).real() > 0 && eigvals(1).real() > 0)*/ {
          // source
          non_saddles.push_back(cp);
          vbo_point_data.push_back(
              {{float(cp(0)), float(cp(1))}, {.0f, .0f, 1.0f}});
        }
      } else {
         if (eigvals(0).real() < 0 && eigvals(1).real() < 0) {
          // swirling sink
          non_saddles.push_back(cp);
          vbo_point_data.push_back(
              {{float(cp(0)), float(cp(1))}, {1.0f, .25f, .25f}});
        } else /*if (eigvals(0).real() > 0 && eigvals(1).real() > 0)*/ {
          // swirling source
          non_saddles.push_back(cp);
          vbo_point_data.push_back(
              {{float(cp(0)), float(cp(1))}, {.25f, .25f, 1.0f}});
        }
      }
    }

    // find boundary switch points
    auto bsps = find_boundary_switch_points(v);
    for (const auto& bsp : bsps) {
        vbo_point_data.push_back(
            {{float(bsp(0)), float(bsp(1))}, {1.0f, 1.0f, 1.0f}});
            //{{float(bsp(0)), float(bsp(1))}, {1.0f, 0.863f, 0.678f}});
        ibo_point_data.push_back(ibo_point_data.size());
    }

    // upload critical points to gpu
    for (size_t i = 0; i < cps.size(); ++i) {
      ibo_point_data.push_back(ibo_point_data.size());
    }
    gpu_point_data_t gpu_point_data{vbo_point_data, ibo_point_data};
    gpu_point_data.setup_vao();

    // create separatrices
    std::vector<integral_curve_t> separatrices;

    for (size_t i = 0; i < saddles.size(); ++i) {
      const auto& saddle  = saddles[i];
      const auto& eigvecs = saddle_eig_vecs[i];
      const auto& eigvals = saddle_eig_vals[i];
      for (size_t i = 0; i < 2; ++i) {
        double       tau = 10;
        const double eps = 1e-7;
        if (eigvals(i) < 0) { tau = -tau; }
        separatrices.push_back(rk43.integrate(
            vn, saddle + normalize(eigvecs.col(i)) * eps, 0, tau));
        separatrices.push_back(rk43.integrate(
            vn, saddle - normalize(eigvecs.col(i)) * eps, 0, tau));
      }
    }
    for (const auto& bsp : bsps) {
      double tau = 10;
      auto a = J(bsp) * v(bsp);
      bool integrate = true;
      
      if (bsp(0) == v.sampler().dimension(0).front()) {
        if (a(0) < 0) { integrate = false; }
      }
      else if (bsp(0) == v.sampler().dimension(0).back()) {
        if (a(0) > 0) { integrate = false; }
      }
      else if (bsp(1) == v.sampler().dimension(1).front()) {
        if (a(1) < 0) { integrate = false; }
      }
      else if (bsp(1) == v.sampler().dimension(1).back()) {
        if (a(1) > 0) { integrate = false; }
      }

      if (integrate) {
        separatrices.push_back(rk43.integrate(vn, bsp, 0, tau));
        separatrices.push_back(rk43.integrate(vn, bsp, 0, -tau));
      }
    }
    // upload separatrices to gpu
    gpu_line_data_t::vbo_data_vec vbo_line_data;
    gpu_line_data_t::ibo_data_vec ibo_line_data;
    size_t line_cnt = 0;
    for (const auto& separatrix : separatrices) {
      for (size_t i = 0; i < separatrix.num_vertices(); ++i) {
        vbo_line_data.push_back(
            {static_cast<float>(separatrix.vertex_at(i)(0)),
             static_cast<float>(separatrix.vertex_at(i)(1))});
      }
      for (size_t i = 0; i < separatrix.num_vertices() - 1; ++i) {
        ibo_line_data.push_back(line_cnt);
        ibo_line_data.push_back(++line_cnt);
      }
      ++line_cnt;
    }
    gpu_line_data_t gpu_line_data{vbo_line_data, ibo_line_data};
    gpu_line_data.setup_vao();

    // create shader
    orthographiccamera cam(v.sampler().front(0), v.sampler().back(0),
                           v.sampler().front(1), v.sampler().back(1), -10, 10,
                           0, 0, lic_res(0), lic_res(1));

    framebuffer fbo{lic_tex};
    fbo.bind();
    yavin::gl::viewport(cam.viewport());
    {
      line_shader shader;
      shader.bind();
      shader.set_projection(cam.projection_matrix());
      yavin::gl::line_width(3);
      gpu_line_data.draw_lines();
    }
    {
      point_shader shader;
      shader.bind();
      shader.set_projection(cam.projection_matrix());
      gpu_point_data.draw_points();
    }
    fbo.unbind();

    // write critical points
    std::string cps_filename{filename};
    auto        dotpos = cps_filename.find_last_of(".");
    cps_filename.replace(dotpos, 4, "_critical_points.vtk");
    write_points(cps, cps_filename);

    // write lic
    std::string lic_filename{filename};
    dotpos = lic_filename.find_last_of(".");
    lic_filename.replace(dotpos, 4, "_skeleton.png");
    lic_tex.write_png(lic_filename);
  }
}
