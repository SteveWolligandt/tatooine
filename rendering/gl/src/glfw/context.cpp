#include <yavin/glfw/context.h>
//==============================================================================
namespace yavin::glfw{
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
}  // namespace yavin::glfw
//==============================================================================
