#include "line_shader.h"
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
line_shader::line_shader() {
  add_stage<yavin::vertexshader>(std::string{vertpath});
  add_stage<yavin::fragmentshader>(std::string{fragpath});
  create();
}
void line_shader::set_modelview_matrix(const yavin::mat4& modelview) {
  set_uniform("modelview", modelview);
}
void line_shader::set_projection_matrix(const yavin::mat4& projmat) {
  set_uniform("projection", projmat);
}
//==============================================================================
}
//==============================================================================
