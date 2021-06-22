#include <tatooine/gl/glfw/context.h>
//==============================================================================
namespace tatooine::gl::glfw{
//==============================================================================
context::context() {
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  m_glfw_window = glfwCreateWindow(internal_resolution, internal_resolution, "",
                                   nullptr, nullptr);
}
//------------------------------------------------------------------------------
context::context(base& parent) {
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  m_glfw_window = glfwCreateWindow(internal_resolution, internal_resolution, "",
                                   nullptr, parent.get());
}
//==============================================================================
}  // namespace tatooine::gl::glfw
//==============================================================================
