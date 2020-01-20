#ifndef TATOOINE_STEADIFICATION_SHADERS_H
#define TATOOINE_STEADIFICATION_SHADERS_H

//==============================================================================
#include <yavin>
#include <string>

//==============================================================================
namespace tatooine::steadification {
//==============================================================================
struct vert_frag_shader : yavin::shader {
  vert_frag_shader(const std::string& vert, const std::string& frag);
  void set_projection(const glm::mat4&);
  void set_modelview(const glm::mat4&);
};
//==============================================================================
struct comp_shader : yavin::shader {
  comp_shader(const std::string& comp);
  void dispatch(GLuint w, GLuint h);
};
//==============================================================================
/// renders vector field, tau and position in three render targets
struct ssf_rasterization_shader : vert_frag_shader {
  ssf_rasterization_shader();
};
//==============================================================================
struct domain_coverage_shader : comp_shader {
  domain_coverage_shader();
};
//==============================================================================
struct ll_to_pos_shader : comp_shader {
  ll_to_pos_shader();
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================

#endif
