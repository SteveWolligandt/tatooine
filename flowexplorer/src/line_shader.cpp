#include <tatooine/flowexplorer/line_shader.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
auto line_shader::get() -> line_shader& {
  static line_shader s;
  return s;
}
line_shader::line_shader() {
  add_stage<rendering::gl::vertexshader>(std::string{vertpath});
  add_stage<rendering::gl::fragmentshader>(std::string{fragpath});
  create();
}
auto line_shader::set_modelview_matrix(const tatooine::mat4f& modelview)
    -> void {
  set_uniform_mat4("modelview", modelview.data_ptr());
}
auto line_shader::set_projection_matrix(const tatooine::mat4f& projmat)
    -> void {
  set_uniform_mat4("projection", projmat.data_ptr());
}
auto line_shader::set_color(GLfloat r, GLfloat g, GLfloat b, GLfloat a)
    -> void {
  set_uniform("color", r, g, b, a);
}
//==============================================================================
}
//==============================================================================
