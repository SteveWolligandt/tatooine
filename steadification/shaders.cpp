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
void vert_frag_shader::set_projection(const yavin::mat4& projection) {
  set_uniform("projection", projection);
}
//------------------------------------------------------------------------------
void vert_frag_shader::set_modelview(const yavin::mat4& modelview) {
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
//------------------------------------------------------------------------------
void ssf_rasterization_shader::set_linked_list_size(unsigned int n) {
  set_uniform("ll_size", n);
}

//==============================================================================
// ll_to_pos
//==============================================================================
ll_to_curvature_shader::ll_to_curvature_shader()
    : comp_shader{"ll_to_curvature_tex.comp"} {}

//==============================================================================
// fragment_count
//==============================================================================
fragment_count_shader::fragment_count_shader()
    : vert_frag_shader{"ssf_rasterization.vert", "fragment_count.frag"} {}

//==============================================================================
// weight_single_pathsurface
//==============================================================================
weight_single_pathsurface_shader::weight_single_pathsurface_shader()
    : comp_shader{"weight_single_pathsurface.comp"} {}
//------------------------------------------------------------------------------
void weight_single_pathsurface_shader::set_linked_list_size(unsigned int n) {
  set_uniform("ll_size", n);
}
//==============================================================================
// weight_dual_pathsurface
//==============================================================================
weight_dual_pathsurface_shader::weight_dual_pathsurface_shader()
    : comp_shader{"weight_dual_pathsurface.comp"} {}
//------------------------------------------------------------------------------
void weight_dual_pathsurface_shader::set_linked_list0_size(unsigned int n) {
  set_uniform("ll0_size", n);
}
//------------------------------------------------------------------------------
void weight_dual_pathsurface_shader::set_linked_list1_size(unsigned int n) {
  set_uniform("ll1_size", n);
}
//==============================================================================
// combine_rasterizations
//==============================================================================
combine_rasterizations_shader::combine_rasterizations_shader()
    : comp_shader{"combine_rasterizations.comp"} {}
//------------------------------------------------------------------------------
void combine_rasterizations_shader::set_linked_list0_size(unsigned int n) {
  set_uniform("ll0_size", n);
}
//------------------------------------------------------------------------------
void combine_rasterizations_shader::set_linked_list1_size(unsigned int n) {
  set_uniform("ll1_size", n);
}
//==============================================================================
// coverage
//==============================================================================
coverage_shader::coverage_shader() : comp_shader{"coverage.comp"} {}
//------------------------------------------------------------------------------
void coverage_shader::set_linked_list_size(unsigned int n) {
  set_uniform("ll_size", n);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
