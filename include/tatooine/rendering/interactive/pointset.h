#ifndef TATOOINE_RENDERING_INTERACTIVE_POINTSET_H
#define TATOOINE_RENDERING_INTERACTIVE_POINTSET_H
//==============================================================================
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/shader.h>
#include <tatooine/linspace.h>
#include <tatooine/pointset.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/interactive/renderer.h>
#include <tatooine/rendering/interactive/interactively_renderable.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
template <floating_point Real>
struct renderer<tatooine::pointset<Real, 2>> {
  using renderable_type = tatooine::pointset<Real, 2>;
  //==============================================================================
  struct shader : gl::shader {
    //------------------------------------------------------------------------------
    static constexpr std::string_view vertex_shader =
        "#version 330 core\n"
        "layout (location = 0) in vec2 position;\n"
        "uniform mat4 projection_matrix;\n"
        "uniform mat4 view_matrix;\n"
        "void main() {\n"
        "  gl_Position = projection_matrix * view_matrix *\n"
        "                vec4(position, 0, 1);\n"
        "}\n";
    //------------------------------------------------------------------------------
    static constexpr std::string_view fragment_shader =
        "#version 330 core\n"
        "uniform vec4 color;\n"
        "out vec4 out_color;\n"
        "void main() {\n"
        "  out_color = color;"
        "}\n";
    //------------------------------------------------------------------------------
    static auto get() -> auto& {
      static auto s = shader{};
      return s;
    }
    //------------------------------------------------------------------------------
   private:
    //------------------------------------------------------------------------------
    shader() {
      add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
      add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
      create();
      set_color(0, 0, 0);
      set_projection_matrix(Mat4<GLfloat>::eye());
      set_view_matrix(Mat4<GLfloat>::eye());
    }
    //------------------------------------------------------------------------------
   public:
    //------------------------------------------------------------------------------
    auto set_color(GLfloat const r, GLfloat const g, GLfloat const b,
                   GLfloat const a = 1) -> void {
      set_uniform("color", r, g, b, a);
    }
    //------------------------------------------------------------------------------
    auto set_projection_matrix(Mat4<GLfloat> const& P) -> void {
      set_uniform_mat4("projection_matrix", P.data().data());
    }
    //------------------------------------------------------------------------------
    auto set_view_matrix(Mat4<GLfloat> const& V) -> void {
      set_uniform_mat4("view_matrix", V.data().data());
    }
  };
  static auto set_projection_matrix(Mat4<GLfloat> const& P) {
    shader::get().set_projection_matrix(P);
  }
  static auto set_view_matrix(Mat4<GLfloat> const& V) {
    shader::get().set_view_matrix(V);
  }
  //==============================================================================
  int                            point_size = 1;
  Vec4<GLfloat>                  color      = {0, 0, 0, 1};
  gl::indexeddata<Vec2<GLfloat>> geometry;
  vec2d                          cursor_pos;
  //==============================================================================
  renderer(renderable_type const& ps) {
    {
      geometry.vertexbuffer().resize(ps.vertices().size());
      auto m = geometry.vertexbuffer().wmap();
      for (auto v : ps.vertices()) {
        m[v.index()] = Vec2<GLfloat>{ps[v]};
      }
    }
    {
      geometry.indexbuffer().resize(ps.vertices().size());
      auto m = geometry.indexbuffer().wmap();
      for (auto v : ps.vertices()) {
        m[v.index()] = v.index();
      }
    }
  }
  //==============================================================================
  auto properties(renderable_type const& /*ps*/) {
    ImGui::Text("Pointset");
    ImGui::DragInt("Point Size", &point_size, 1, 1, 20);
    ImGui::ColorEdit4("Color", color.data().data());
  }
  //==============================================================================
  auto render() {
    shader::get().bind();
    shader::get().set_color(color(0), color(1), color(2), color(3));
    gl::point_size(point_size);
    geometry.draw_points();
  }

  auto on_cursor_moved(double const x, double const y) { cursor_pos = {x, y}; }
  auto on_button_pressed(gl::button) {}
  auto on_button_released(gl::button) {}
};
static_assert(interactively_renderable<renderer<pointset2>>);
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
