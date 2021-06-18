#ifndef TATOOINE_RENDERING_LINE_H
#define TATOOINE_RENDERING_LINE_H
//==============================================================================
#include <tatooine/line.h>
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/vec.h>
#include <tatooine/rendering/gl/indexeddata.h>
#include <tatooine/rendering/gl/shader.h>
//==============================================================================
namespace tatooine::rendering  {
//==============================================================================
struct line_shader : rendering::gl::shader {
  line_shader() {
    add_stage<rendering::gl::vertexshader>(rendering::gl::shadersource{
        "#version 330 core\n"
        "layout(location = 0) in vec3 pos;\n"
        "uniform mat4 projection_matrix;\n"
        "uniform mat4 modelview_matrix;\n"
        "//------------------------------------------------------------------\n"
        "void main() {\n"
        "  gl_Position = projection_matrix * modelview_matrix * vec4(pos, 1);\n"
        "}\n"});
    add_stage<rendering::gl::fragmentshader>(rendering::gl::shadersource{
        "#version 330 core\n"
        "layout(location = 0) out vec4 fragout;\n"
        "//------------------------------------------------------------------\n"
        "void main() {\n"
        "  fragout = vec4(0, 0, 0, 1);\n"
        "}\n"});
    create();
  }
  auto set_projection_matrix(mat4f const& P) {
    set_uniform_mat4("projection_matrix", P.data_ptr());
  }
  auto set_modelview_matrix(mat4f const& MV) {
    set_uniform_mat4("modelview_matrix", MV.data_ptr());
  }
};
template <typename Real>
auto to_gpu(line<Real, 3> const& l) {
  rendering::gl::indexeddata<vec3f> gpu_data;
  gpu_data.vertexbuffer().resize(l.num_vertices());
  {
    auto gpu_mapping = gpu_data.vertexbuffer().wmap();
    for (auto const v : l.vertices()) {
      auto const& x = l[v];
      gpu_mapping[v.i] = x;
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
  return gpu_data;
}
//------------------------------------------------------------------------------
template <typename Real>
auto to_gpu(std::vector<line<Real, 3>> const& lines) {
  rendering::gl::indexeddata<vec3f> gpu_data;
  size_t                    num_vertices = 0;
  size_t                    num_indices = 0;
  for (auto const& l : lines) {
    num_vertices += l.num_vertices();
    num_indices += (l.num_vertices() - 1) * 2;
  }
  gpu_data.vertexbuffer().resize(num_vertices);
  gpu_data.indexbuffer().resize(num_indices);
  {
    auto gpu_mapping = gpu_data.vertexbuffer().wmap();
    size_t i           = 0;
    for (auto const& l : lines) {
      for (auto const v : l.vertices()) {
        auto const& x    = l[v];
        gpu_mapping[i++] = x;
      }
    }
  }
  gpu_data.indexbuffer().resize(num_indices);
  {
    auto gpu_mapping = gpu_data.indexbuffer().wmap();
    size_t i           = 0;
    size_t k = 0;
    for (auto const& l : lines) {
      for (size_t j = 0; j < l.num_vertices() - 1; ++i, ++j) {
        gpu_mapping[k++] = i;
        gpu_mapping[k++] = i + 1;
      }
      ++i;
    }
  }
  return gpu_data;
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
auto interactive(std::vector<line<Real, N>> const& lines) {
  auto win = first_person_window{};
  auto shader = line_shader{};
  shader.bind();
  auto       aabb       = axis_aligned_bounding_box<Real, N>{};
  for (auto const& l : lines) {
    for (auto const v : l.vertices()) {
      aabb += l[v];
    }
  }
  auto       gpu_data   = to_gpu(lines);
  auto const center_pos = aabb.center();
  win.camera_controller().look_at(center_pos + vec{2, 2, 2}, center_pos);
  shader.bind();
  shader.set_projection_matrix(win.camera_controller().projection_matrix());
  rendering::gl::clear_color(255, 255, 255, 255);
  bool my_tool_active = true;
  win.render_loop([&](auto const dt) {
      rendering::gl::clear_color_depth_buffer();
    shader.set_modelview_matrix(win.camera_controller().view_matrix());
    shader.set_projection_matrix(win.camera_controller().projection_matrix());
    gpu_data.draw_lines();
  });
}
//==============================================================================
template <typename Real, size_t N>
auto interactive(line<Real, N> const& l) {
  auto win = first_person_window{};
  auto shader = line_shader{};
  shader.bind();
  auto       aabb       = axis_aligned_bounding_box<Real, N>{};
  for (auto const v : l.vertices()) {
    aabb += l[v];
  }
  auto       gpu_data   = to_gpu(l);
  auto const center_pos = aabb.center();
  win.camera_controller().look_at(center_pos + vec{2, 2, 2}, center_pos);
  shader.bind();
  shader.set_projection_matrix(win.camera_controller().projection_matrix());
  rendering::gl::clear_color(255, 255, 255, 255);
  bool my_tool_active = true;
  win.render_loop([&](auto const dt) {
      rendering::gl::clear_color_depth_buffer();
    shader.set_modelview_matrix(win.camera_controller().view_matrix());
    shader.set_projection_matrix(win.camera_controller().projection_matrix());
    gpu_data.draw_lines();
  });
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
