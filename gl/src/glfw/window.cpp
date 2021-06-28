#include <tatooine/gl/glfw/api.h>
#include <tatooine/gl/glfw/window.h>

#include <stdexcept>
//==============================================================================
namespace tatooine::gl::glfw {
//==============================================================================
window::window(size_t const width, size_t const height,
               std::string const& title) {
  api::get();
  glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
  m_glfw_window =
      glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
  if (!m_glfw_window) {
    throw std::runtime_error{"[GLFW] Could not create window."};
  }
  make_current();
  glfwSwapInterval(1);
  glfwSetWindowUserPointer(m_glfw_window, this);
  glfwSetWindowCloseCallback(m_glfw_window, api::on_close);
  glfwSetWindowSizeCallback(m_glfw_window, api::on_resize);
  glfwSetCursorPosCallback(m_glfw_window, api::on_cursor_moved);
  glfwSetMouseButtonCallback(m_glfw_window, api::on_button);
  glfwSetScrollCallback(m_glfw_window, api::on_mouse_wheel);
  glfwSetKeyCallback(m_glfw_window, api::on_key);
}
//------------------------------------------------------------------------------
auto window::should_close() const -> bool {
  return glfwWindowShouldClose(m_glfw_window);
}
//------------------------------------------------------------------------------
auto window::swap_buffers() -> void { glfwSwapBuffers(m_glfw_window); }
//==============================================================================
}  // namespace tatooine::gl::glfw
//==============================================================================
