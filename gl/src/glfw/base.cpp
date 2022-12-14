#include <tatooine/gl/glfw/base.h>
//==============================================================================
namespace tatooine::gl::glfw {
//==============================================================================
base::base() {
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
}
//------------------------------------------------------------------------------
base::~base() { glfwDestroyWindow(m_glfw_window); }
//----------------------------------------------------------------------------
auto base::get() -> GLFWwindow* { return m_glfw_window; }
auto base::get() const -> GLFWwindow const* { return m_glfw_window; }
//------------------------------------------------------------------------------
auto base::make_current() -> void {
  glfwMakeContextCurrent(m_glfw_window);
  gladLoadGL();
}
//------------------------------------------------------------------------------
auto base::release() -> void { glfwMakeContextCurrent(nullptr); }
//------------------------------------------------------------------------------
auto base::get_window_size(int* w, int* h) -> void {
  glfwGetWindowSize(m_glfw_window, w, h);
}
//==============================================================================
}  // namespace tatooine::gl::glfw
//==============================================================================
