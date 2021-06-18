#include <tatooine/flowexplorer/point_shader.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
point_shader::point_shader() {
  add_stage<rendering::gl::vertexshader>(vertpath);
  add_stage<rendering::gl::fragmentshader>(fragpath);
  create();
}
//------------------------------------------------------------------------------
auto point_shader::get() -> point_shader& {
  static point_shader instance;
  return instance;
}
//------------------------------------------------------------------------------
auto point_shader::set_modelview_matrix(
    const tatooine::mat<float, 4, 4>& modelview) -> void {
  set_uniform_mat4("modelview", modelview.data_ptr());
}
//------------------------------------------------------------------------------
auto point_shader::set_projection_matrix(
    const tatooine::mat<float, 4, 4>& projmat) -> void {
  set_uniform_mat4("projection", projmat.data_ptr());
}
//------------------------------------------------------------------------------
auto point_shader::set_color(GLfloat r, GLfloat g, GLfloat b, GLfloat a)
    -> void {
  set_uniform("color", r, g, b, a);
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
