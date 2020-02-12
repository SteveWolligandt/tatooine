#include <tatooine/critical_points.h>
#include <tatooine/gpu/lic.h>
//#include <tatooine/diff.h>
#include <tatooine/vtk_legacy.h>
#include <yavin/context.h>
#include <yavin/framebuffer.h>
#include <yavin/orthographiccamera.h>
#include <yavin/vertexbuffer.h>
#include <yavin/vertexarray.h>
#include <yavin/indexbuffer.h>
#include <yavin/glwrapper.h>
#include <yavin/indexeddata.h>

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
        "  frag = vec4(1,0,0,1);\n"
        "}\n",
        shaderstageparser::SOURCE);
    create();
  }
  void set_projection(const glm::mat4x4& p) { set_uniform("projection", p); }
};
//==============================================================================
void write_points(std::vector<tatooine::vec<double, 2>>& points,
                  const std::string&           path) {
  using namespace tatooine;
  using namespace boost;
  using namespace adaptors;
  std::vector<vec<double, 3>> points3(points.size());
  copy(points | transformed([](const auto& p2) {
         return vec{p2(0), p2(1), 0};
       }),
       points3.begin());
  vtk::legacy_file_writer writer(path, vtk::POLYDATA);
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
  for (const auto& filename : filenames) {
    std::cerr << "[" << filename << "]\n";
    // read file
    sampled_field<grid_sampler<double, 2, vec<double, 2>, linear, linear>, double,
                  2, 2>
        v;
    v.sampler().read_vtk_scalars(std::string{filename}, "vector_field");

    const vec<size_t, 2> lic_res{v.sampler().size(0) * 30, v.sampler().size(1) * 30};
    auto lic_tex = gpu::lic(v.sampler(), lic_res, 100, 0.00005, {256, 256});

    // calculate critical points
    auto cps = find_critical_points(v.sampler());
    
    // specify critical points
    std::vector <vec<double, 2>> saddles;
    //auto J = diff(v);
    //for (const auto& cp : cps) {
    //  auto j = J(cp);
    //  auto [eigvals, eigves] = eigenvectors(j);
    //}

    // upload critical points to gpu
    using gpu_data_t = indexeddata<yavin::vec2>;
    gpu_data_t::vbo_data_vec vbo_data;
    gpu_data_t::ibo_data_vec ibo_data;
    for (size_t i = 0; i < cps.size(); ++i) {
      vbo_data.push_back({float(cps[i](0)), float(cps[i](1))});
    }
    for (size_t i = 0; i < cps.size(); ++i) { ibo_data.push_back(i); }
    gpu_data_t gpu_data{vbo_data, ibo_data};
    gpu_data.setup_vao();

    // create shader
    orthographiccamera cam(v.sampler().front(0), v.sampler().back(0), v.sampler().front(1),
                           v.sampler().back(1), -10, 10, 0, 0, lic_res(0),
                           lic_res(1));
    point_shader       shader;
    shader.bind();
    shader.set_projection(cam.projection_matrix());

    framebuffer fbo{lic_tex};
    fbo.bind();
    yavin::gl::viewport(cam.viewport());
    yavin::gl::point_size(10);
    gpu_data.draw_points();
    fbo.unbind();

    // write critical points
    // for (auto cp : cps) { std::cerr << "  " << cp << '\n'; }
    std::string cps_filename{filename};
    auto        dotpos = cps_filename.find_last_of(".");
    cps_filename.replace(dotpos, 4, "_critical_points.vtk");
    write_points(cps, cps_filename);

    // write lic
    std::string lic_filename{filename};
    dotpos = lic_filename.find_last_of(".");
    lic_filename.replace(dotpos, 4, "_lic.png");
    lic_tex.write_png(lic_filename);
    }
}
