#ifndef TATOOINE_GL_GLFW_CONTEXT_H
#define TATOOINE_GL_GLFW_CONTEXT_H
//==============================================================================
#include <tatooine/gl/glfw/base.h>
//==============================================================================
namespace tatooine::gl::glfw {
//==============================================================================
struct context : base {
  static constexpr size_t internal_resolution = 64;
  //============================================================================
  context();
  context(base& parent);
};
//==============================================================================
}  // namespace tatooine::gl::glfw
//==============================================================================
#endif
