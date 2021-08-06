#ifndef TATOOINE_FLOWEXPLORER_NODES_RECTILINEAR_GRID_VERTEX_PROPERTY_RENDERER_H
#define TATOOINE_FLOWEXPLORER_NODES_RECTILINEAR_GRID_VERTEX_PROPERTY_RENDERER_H
//==============================================================================
#include <tatooine/color_scales/viridis.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/rectilinear_grid.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct rectilinear_grid_vertex_property_renderer
    : renderable<rectilinear_grid_vertex_property_renderer> {
  struct shader_t : gl::shader {
   private:
    shader_t() {
      add_stage<gl::vertexshader>(gl::shadersource{
          "#version 330\n"
          "uniform mat4 projection;\n"
          "uniform mat4 modelview;\n"
          "layout(location = 0) in vec3 position;\n"
          "void main() {\n"
          "  gl_Position = projection * modelview * vec4(position, 1);\n"
          "}\n"});
      add_stage<gl::fragmentshader>(gl::shadersource{
          "#version 330\n"
          "layout(location = 0) out vec4 frag_out;\n"
          "void main() {\n"
          "  frag_out = vec4(0,0,0,1);\n"
          "}\n"});
      create();
    }

   public:
    static auto get() -> auto& {
      static shader_t shader{};
      return shader;
    }
    auto set_modelview_matrix(const tatooine::mat4f& modelview) -> void {
      set_uniform_mat4("modelview", modelview.data_ptr());
    }
    auto set_projection_matrix(const tatooine::mat4f& projmat) -> void {
      set_uniform_mat4("projection", projmat.data_ptr());
    }
  };
  //----------------------------------------------------------------------------
  ui::input_pin*         m_rect_grid_in;
   gl::indexeddata<vec3f> m_gpu_geometry;
  // gl::tex2r32f           m_data;
   //gl::tex1rgb32f         m_color_scale;
  //----------------------------------------------------------------------------
  rectilinear_grid_vertex_property_renderer(flowexplorer::scene& s)
    : renderable<rectilinear_grid_vertex_property_renderer>{"Vertex Property Renderer", s},
    m_rect_grid_in{&this->template insert_input_pin<nonuniform_rectilinear_grid2>("Grid")}
  //,      m_color_scale{color_scales::viridis{}.to_gpu_tex()}
  {
    create_indexed_data();
  }
  //----------------------------------------------------------------------------
  auto render(mat4f const& P, mat4f const& V) -> void override {
    if (m_rect_grid_in->is_linked()) {
      update_vbo_data();
      auto& shader = shader_t::get();
      shader.bind();
      shader.set_projection_matrix(P);
      shader.set_modelview_matrix(V);
      m_gpu_geometry.draw_triangles();
    }
  }
  //----------------------------------------------------------------------------
  auto is_transparent() const -> bool override { return false; }
  //----------------------------------------------------------------------------
  auto update_vbo_data() -> void {
    auto& grid = m_rect_grid_in->get_linked_as<nonuniform_rectilinear_grid2>();
    {
      auto vbomap = m_gpu_geometry.vertexbuffer().map();
      vbomap[0] =
          vec3f{float(grid.front<0>()), float(grid.front<1>()), float(0)};
      vbomap[1] =
          vec3f{float(grid.back<0>()), float(grid.front<1>()), float(0)};
      vbomap[2] =
          vec3f{float(grid.front<1>()), float(grid.back<1>()), float(0)};
      vbomap[3] = vec3f{float(grid.back<0>()), float(grid.back<1>()), float(0)};
    }
  }
  //----------------------------------------------------------------------------
  auto create_indexed_data() -> void {
    m_gpu_geometry.vertexbuffer().resize(4);
    m_gpu_geometry.indexbuffer().resize(6);
    m_gpu_geometry.indexbuffer() = {0, 1, 3, 2, 1, 3};
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(
    tatooine::flowexplorer::nodes::rectilinear_grid_vertex_property_renderer);
#endif
