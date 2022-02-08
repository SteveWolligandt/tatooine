#ifndef TATOOINE_RENDERING_INTERACTIVE_SHADERS_H
#define TATOOINE_RENDERING_INTERACTIVE_SHADERS_H
//==============================================================================
#include <tatooine/gl/shader.h>
//==============================================================================
namespace tatooine::rendering::interactive::shaders {
//==============================================================================
struct colored_pass_through_2d_without_matrices : gl::shader {
  //------------------------------------------------------------------------------
  static constexpr std::string_view vertex_shader =
      "#version 330 core\n"
      "layout (location = 0) in vec2 position;\n"
      "void main() {\n"
      "  gl_Position = vec4(position, 0, 1);\n"
      "}\n";
  //------------------------------------------------------------------------------
  static constexpr std::string_view fragment_shader =
      "#version 330 core\n"
      "uniform vec4 color;\n"
      "out vec4 out_color;\n"
      "void main() {\n"
      "  out_color = color;\n"
      "}\n";
  //------------------------------------------------------------------------------
  static auto get() -> auto& {
    static auto s = colored_pass_through_2d_without_matrices{};
    return s;
  }
  //------------------------------------------------------------------------------
 private:
  //------------------------------------------------------------------------------
  colored_pass_through_2d_without_matrices() {
    add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
    add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
    create();
    set_color(0, 0, 0);
  }
  //------------------------------------------------------------------------------
 public:
  //------------------------------------------------------------------------------
  auto set_color(GLfloat r, GLfloat g, GLfloat b, GLfloat a = 1) -> void {
    set_uniform("color", r, g, b, a);
  }
};
//==============================================================================
struct colored_pass_through_2d : gl::shader {
  //------------------------------------------------------------------------------
  static constexpr std::string_view vertex_shader =
      "#version 330 core\n"
      "uniform mat4 projection_matrix;\n"
      "uniform mat4 model_view_matrix;\n"
      "layout (location = 0) in vec2 position;\n"
      "void main() {\n"
      "  gl_Position = projection_matrix * \n"
      "                model_view_matrix * \n"
      "                vec4(position, 0, 1);\n"
      "}\n";
  //------------------------------------------------------------------------------
  static constexpr std::string_view fragment_shader =
      "#version 330 core\n"
      "uniform vec4 color;\n"
      "out vec4 out_color;\n"
      "void main() {\n"
      "  out_color = color;\n"
      "}\n";
  //------------------------------------------------------------------------------
  static auto get() -> auto& {
    static auto s = colored_pass_through_2d{};
    return s;
  }
  //------------------------------------------------------------------------------
 private:
  //------------------------------------------------------------------------------
  colored_pass_through_2d() {
    add_stage<gl::vertexshader>(gl::shadersource{vertex_shader});
    add_stage<gl::fragmentshader>(gl::shadersource{fragment_shader});
    create();
    set_color(0, 0, 0);
    set_model_view_matrix(Mat4<GLfloat>::eye());
    set_projection_matrix(Mat4<GLfloat>::eye());
  }
  //------------------------------------------------------------------------------
 public:
  //------------------------------------------------------------------------------
  auto set_color(GLfloat r, GLfloat g, GLfloat b, GLfloat a = 1) -> void {
    set_uniform("color", r, g, b, a);
  }
  //------------------------------------------------------------------------------
  auto set_model_view_matrix(Mat4<GLfloat> const& MV) -> void {
    set_uniform_mat4("model_view_matrix", MV.data().data());
  }
  //------------------------------------------------------------------------------
  auto set_projection_matrix(Mat4<GLfloat> const& P) -> void {
    set_uniform_mat4("projection_matrix", P.data().data());
  }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive::shaders
//==============================================================================
#endif
