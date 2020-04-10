#include <shaders.h>

//==============================================================================
using namespace yavin;
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
// vert_frag_shader
//==============================================================================
vert_frag_shader::vert_frag_shader(const std::string_view& vert,
                                   const std::string_view& frag) {
  add_stage<vertexshader>(std::string{vert});
  add_stage<fragmentshader>(std::string{frag});
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
comp_shader::comp_shader(const std::string_view& comp) {
  add_stage<computeshader>(std::string{comp});
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
    : vert_frag_shader{vert_path, frag_path} {}
//------------------------------------------------------------------------------
void ssf_rasterization_shader::set_count(GLboolean c) {
  set_uniform("count", c);
}
//------------------------------------------------------------------------------
void ssf_rasterization_shader::set_width(GLuint w) {
  set_uniform("width", w);
}
//------------------------------------------------------------------------------
void ssf_rasterization_shader::set_render_index(GLuint render_index) {
  set_uniform("render_index", render_index);
}
//------------------------------------------------------------------------------
void ssf_rasterization_shader::set_layer(GLuint layer) {
  set_uniform("layer", layer);
}
//==============================================================================
// tex_rasterization_to_buffer_shader
//==============================================================================
tex_rasterization_to_buffer_shader::tex_rasterization_to_buffer_shader()
    : comp_shader{comp_path} {}
//==============================================================================
// lic
//==============================================================================
lic_shader::lic_shader() : comp_shader{comp_path} {
  set_v_tex_bind_point(0);
  set_noise_tex_bind_point(1);
  set_color_scale_bind_point(2);
}
void lic_shader::set_domain_min(GLfloat x, GLfloat y) {
  set_uniform("domain_min", x, y);
}
void lic_shader::set_domain_max(GLfloat x, GLfloat y) {
  set_uniform("domain_max", x, y);
}
void lic_shader::set_backward_tau(GLfloat btau) {
  set_uniform("btau", btau);
}
void lic_shader::set_forward_tau(GLfloat ftau) {
  set_uniform("ftau", ftau);
}
void lic_shader::set_v_tex_bind_point(GLint b) {
  set_uniform("v_tex", b);
}
void lic_shader::set_noise_tex_bind_point(GLint b) {
  set_uniform("noise_tex", b);
}
void lic_shader::set_color_scale_bind_point(GLint b) {
  set_uniform("color_scale", b);
}
void lic_shader::set_num_samples(GLuint n) {
  set_uniform("num_samples", n);
}
void lic_shader::set_stepsize(GLfloat s) {
  set_uniform("stepsize", s);
}

//==============================================================================
// ll_to_v
//==============================================================================
ll_to_v_shader::ll_to_v_shader() : comp_shader{comp_path} {}

//==============================================================================
// ll_to_curvature
//==============================================================================
ll_to_curvature_shader::ll_to_curvature_shader() : comp_shader{comp_path} {}

//==============================================================================
// fragment_count
//==============================================================================
fragment_count_shader::fragment_count_shader()
    : vert_frag_shader{vert_path, frag_path} {}

//==============================================================================
// seedcurve
//==============================================================================
seedcurve_shader::seedcurve_shader() : vert_frag_shader{vert_path, frag_path} {
  set_color_scale_bind_point(2);
  set_min_t(0);
  set_max_t(1);
}
void seedcurve_shader::set_color_scale_bind_point(GLint b) {
  set_uniform("color_scale", b);
}
void seedcurve_shader::set_min_t(GLfloat t) {
  set_uniform("min_t", t);
}
void seedcurve_shader::set_max_t(GLfloat t) {
  set_uniform("max_t", t);
};

//==============================================================================
// weight_dual_pathsurface
//==============================================================================
weight_dual_pathsurface_shader::weight_dual_pathsurface_shader()
    : comp_shader{comp_path} {}
//------------------------------------------------------------------------------
void weight_dual_pathsurface_shader::set_layer(GLuint n) {
  set_uniform("layer", n);
}
//------------------------------------------------------------------------------
void weight_dual_pathsurface_shader::set_size(GLuint n) {
  set_uniform("size", n);
}
//------------------------------------------------------------------------------
void weight_dual_pathsurface_shader::set_penalty(GLfloat p) {
  set_uniform("penalty", p);
}
//==============================================================================
// combine_rasterizations
//==============================================================================
combine_rasterizations_shader::combine_rasterizations_shader()
    : comp_shader{comp_path} {}
//------------------------------------------------------------------------------
void combine_rasterizations_shader::set_resolution(GLuint w, GLuint h) {
  set_uniform("resolution", w, h);
}
//==============================================================================
// coverage
//==============================================================================
coverage_shader::coverage_shader() : comp_shader{comp_path} {}
//------------------------------------------------------------------------------
void coverage_shader::set_linked_list_size(unsigned int n) {
  set_uniform("ll_size", n);
}
//==============================================================================
// dual_coverage
//==============================================================================
dual_coverage_shader::dual_coverage_shader() : comp_shader{comp_path} {}
//------------------------------------------------------------------------------
void dual_coverage_shader::set_linked_list0_size(unsigned int n) {
  set_uniform("ll0_size", n);
}
//------------------------------------------------------------------------------
void dual_coverage_shader::set_linked_list1_size(unsigned int n) {
  set_uniform("ll1_size", n);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
