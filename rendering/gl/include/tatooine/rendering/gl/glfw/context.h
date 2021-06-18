#ifndef TATOOINE_RENDERING_GL_GLFW_CONTEXT_H
#define TATOOINE_RENDERING_GL_GLFW_CONTEXT_H
//==============================================================================
#include <tatooine/rendering/gl/glfw/base.h>
//==============================================================================
namespace tatooine::rendering::gl::glfw {
//==============================================================================
struct context : base {
  static constexpr size_t internal_resolution = 64;
  //============================================================================
  context();
  context(base& parent);
};
//==============================================================================
}  // namespace tatooine::rendering::gl::glfw
//==============================================================================
#endif
