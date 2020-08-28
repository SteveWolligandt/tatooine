#include "line_shader.h"
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
line_shader::line_shader() {
  add_stage<yavin::vertexshader>(std::string{vertpath});
  add_stage<yavin::fragmentshader>(std::string{fragpath});
  create();
}
void line_shader::set_modelview_matrix(
    const tatooine::mat<float, 4, 4>& modelview) {
  set_uniform_mat4("modelview", modelview.data_ptr());
}
void line_shader::set_projection_matrix(
    const tatooine::mat<float, 4, 4>& projmat) {
  set_uniform_mat4("projection", projmat.data_ptr());
}
void line_shader::set_color(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
  set_uniform("color", r, g, b, a);
}
//==============================================================================
}
//==============================================================================
