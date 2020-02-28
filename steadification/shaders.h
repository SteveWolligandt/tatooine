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
  void set_linked_list_size(unsigned int n);
};
//==============================================================================
struct domain_coverage_shader : comp_shader {
  domain_coverage_shader();
};
//==============================================================================
struct ll_to_curvature_shader : comp_shader {
  ll_to_curvature_shader();
};
//==============================================================================
struct fragment_count_shader : vert_frag_shader {
  fragment_count_shader();
};
//==============================================================================
struct weight_single_pathsurface_shader : comp_shader {
  weight_single_pathsurface_shader();
  void set_linked_list_size(unsigned int n);
};
//==============================================================================
struct weight_dual_pathsurface_shader : comp_shader {
  weight_dual_pathsurface_shader();
  void set_linked_list0_size(unsigned int n);
  void set_linked_list1_size(unsigned int n);
};
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
