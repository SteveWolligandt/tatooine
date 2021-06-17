#ifndef TATOOINE_RENDERING_LINE_H
#define TATOOINE_RENDERING_LINE_H
//==============================================================================
#include <tatooine/line.h>
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/rendering/yavin_interop.h>
#include <tatooine/vec.h>
#include <yavin/indexeddata.h>
#include <yavin/shader.h>
//==============================================================================
namespace tatooine::rendering  {
//==============================================================================
struct line_shader : yavin::shader{
  line_shader() {
    add_stage<yavin::vertexshader>(yavin::shadersource{
      "#version 330 core\n"
      "layout(location = 0) in vec3 pos;\n"
      "uniform mat4 projection_matrix;\n"
      "uniform mat4 modelview_matrix;\n"
      "//--------------------------------------------------------------------\n"
      "void main() {\n"
      "  gl_Position = projection_matrix * modelview_matrix * vec4(pos, 1);\n"
      "}\n"
    });
    add_stage<yavin::fragmentshader>(yavin::shadersource{
      "#version 330 core\n"
      "layout(location = 0) out vec4 fragout;\n"
      "//--------------------------------------------------------------------\n"
      "void main() {\n"
      "  fragout = vec4(0, 0, 0, 1);\n"
      "}\n"
    });
    create();
  }
  auto set_projection_matrix(mat4f const& P) {
    set_uniform_mat4("projection_matrix", P.data_ptr());
  }
  auto set_modelview_matrix(mat4f const& MV) {
    set_uniform_mat4("modelview_matrix", MV.data_ptr());
  }
};
//==============================================================================
template <typename Real, size_t N>
auto interactive(line<Real, N> const& l) {
  auto win = first_person_window{};
  auto shader = line_shader{};
  shader.bind();
  yavin::indexeddata<vec3f> gpu_data;
  gpu_data.vertexbuffer().resize(l.num_vertices());
  {
    auto gpu_mapping = gpu_data.vertexbuffer().wmap();
    for (auto const v : l.vertices()) {
      gpu_mapping[v.i] = l[v];
    }
  }
  gpu_data.indexbuffer().resize((l.num_vertices() - 1) * 2);
  {
    auto gpu_mapping = gpu_data.indexbuffer().wmap();
    for (size_t i = 0; i < l.num_vertices() - 1; ++i) {
      gpu_mapping[i * 2]     = i;
      gpu_mapping[i * 2 + 1] = i + 1;
    }
  }
  shader.bind();
  shader.set_projection_matrix(win.camera_controller().projection_matrix());
    yavin::gl::clear_color(255, 255, 255, 255);
  win.render_loop([&](auto const dt) {
    yavin::clear_color_depth_buffer();
    shader.set_modelview_matrix(win.camera_controller().view_matrix());
    gpu_data.draw_lines();
  });
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
