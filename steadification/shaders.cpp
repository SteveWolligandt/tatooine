#include "shaders.h"

//==============================================================================
using namespace yavin;
//==============================================================================
namespace tatooine::steadification {
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
void comp_shader::dispatch(GLuint w, GLuint h) {
  bind();
  gl::dispatch_compute(w, h, 1);
}
//==============================================================================
// ssf_rasterization_shader
//==============================================================================
ssf_rasterization_shader::ssf_rasterization_shader()
    : vert_frag_shader{"ssf_rasterization.vert",
                       "ssf_rasterization.frag"} {}

//==============================================================================
// domain_coverage_shader
//==============================================================================
domain_coverage_shader::domain_coverage_shader()
    : comp_shader{"domain_coverage.comp"} {}

//==============================================================================
// ll_to_pos
//==============================================================================
ll_to_pos_shader::ll_to_pos_shader() : comp_shader{"ll_to_pos.comp"} {}

//==============================================================================
}  // namespace tatooine
//==============================================================================
