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
  void dispatch2d(GLuint w, GLuint h);
};
//==============================================================================
/// renders vector field and tau in two render targets
struct v_tau_shader : vert_frag_shader {
  v_tau_shader();
};
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
