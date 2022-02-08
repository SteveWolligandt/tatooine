#ifndef TATOOINE_FLOWEXPLORER_NODES_VTK_LINE_READER_H
#define TATOOINE_FLOWEXPLORER_NODES_VTK_LINE_READER_H
//==============================================================================
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/rendering/matrices.h>
#include <tatooine/flowexplorer/line_shader.h>
#include <tatooine/vtk_legacy.h>
#include <tatooine/line.h>
#include <tatooine/gl/indexeddata.h>

#include <mutex>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct vtk_line_reader : renderable<vtk_line_reader> {
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // internal
 private:
  gl::indexeddata<vec3f> m_gpu_data;
  line<real_type, 3>                   m_line3;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // user data
  std::string            m_path;
  std::array<GLfloat, 4> m_line_color{0.0f, 0.0f, 0.0f, 1.0f};
  int                    m_line_width = 1;

 public:
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  vtk_line_reader(flowexplorer::scene& s) : renderable{"VTK Line Reader", s} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  auto render(mat4f const& P, mat4f const& V) -> void override {
    auto& shader = line_shader::get();
    shader.bind();
    shader.set_color(m_line_color[0], m_line_color[1], m_line_color[2],
                     m_line_color[3]);
    shader.set_projection_matrix(P);
    shader.set_modelview_matrix(V);
    gl::line_width(m_line_width);
    m_gpu_data.draw_lines();
   }
   //----------------------------------------------------------------------------
   // setters / getters
   //----------------------------------------------------------------------------
   auto path() -> auto& { return m_path; }
   auto path() const -> auto const& { return m_path; }
   //----------------------------------------------------------------------------
   auto update(std::chrono::duration<double> const& /*dt*/) -> void override {}
   //----------------------------------------------------------------------------
   auto draw_properties() -> bool override {
     bool changed = false;
     if (ImGui::Button("open")) {
       scene().window().open_file_explorer(
           "open vtk line", std::vector{".vtk"}, *this);
     }
     ImGui::SameLine();
     if (ImGui::Button("reload")) {
       read();
     }
     changed |= ImGui::SliderInt("line width", &m_line_width, 1, 50);
     changed |= ImGui::ColorEdit4("line color", m_line_color.data());
     return changed;
   }
   //----------------------------------------------------------------------------
   auto on_path_selected(std::string const& path) -> void override {
     m_path = path;
     read();
   }
   //----------------------------------------------------------------------------
   auto read() -> void {
     m_line3 = m_line3.read_vtk(m_path).front();


     bool insert_seg = false;
     int i = 0;
     m_gpu_data.clear();
     for (auto const& y : m_line3.vertices()) {
       m_gpu_data.vertexbuffer().push_back(vec3f{static_cast<GLfloat>(y(0)),
                                                 static_cast<GLfloat>(y(1)),
                                                 static_cast<GLfloat>(y(2))});
       if (insert_seg) {
         m_gpu_data.indexbuffer().push_back(i - 1);
         m_gpu_data.indexbuffer().push_back(i);
       } else {
         insert_seg = true;
       }
       ++i;
     }
   }
   auto is_transparent() const -> bool override { return m_line_color[3] < 255; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::vtk_line_reader,
    TATOOINE_REFLECTION_INSERT_METHOD(path, path()))
#endif
