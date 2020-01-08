#ifndef SHADERS_H
#define SHADERS_H

//==============================================================================
#include <yavin>
#include <string>

//==============================================================================
namespace tatooine {
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
/// renders vector field, tau and position in three render targets
struct domain_coverage_shader : comp_shader {
  domain_coverage_shader();
};
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
