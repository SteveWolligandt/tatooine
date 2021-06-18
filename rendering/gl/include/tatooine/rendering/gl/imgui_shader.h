#ifndef YAVIN_IMGUI_SHADER_H
#define YAVIN_IMGUI_SHADER_H
//==============================================================================
#include "shader.h"
//==============================================================================
namespace yavin{
//==============================================================================
struct imgui_shader : shader {
  static const std::string_view vert_source;
  static const std::string_view frag_source;
  imgui_shader();
  void set_projection_matrix(std::array<GLfloat, 16> const& p);
  void set_texture_slot(int s);
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
