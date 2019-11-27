#include "shaders.h"

//==============================================================================
using namespace yavin;
//==============================================================================
namespace tatooine {
//==============================================================================
// vert_frag_shader
//==============================================================================
vert_frag_shader::vert_frag_shader(const std::string& vert,
                                   const std::string& frag) {
  add_stage<vertexshader>(vert);
  add_stage<fragmentshader>(frag);
  create();
}
//------------------------------------------------------------------------------
void vert_frag_shader::set_projection(const glm::mat4& projection) {
  set_uniform("projection", projection);
}
//------------------------------------------------------------------------------
void vert_frag_shader::set_modelview(const glm::mat4& modelview) {
  set_uniform("modelview", modelview);
}
//==============================================================================
// comp_shader
//==============================================================================
comp_shader::comp_shader(const std::string& comp) {
  add_stage<computeshader>(comp);
  create();
}
//------------------------------------------------------------------------------
void comp_shader::dispatch2d(GLuint w, GLuint h) {
  bind();
  gl::dispatch_compute(w, h, 1);
}
//==============================================================================
// v_tau_shader
//==============================================================================
v_tau_shader::v_tau_shader()
    : vert_frag_shader{"v_tau_shader.vert", "v_tau_shader.frag"} {}
//==============================================================================
}  // namespace tatooine
//==============================================================================
