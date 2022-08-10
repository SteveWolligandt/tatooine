#include <tatooine/gl/glfw/context.h>
#include <tatooine/gl/glfw/api.h>

#include <stdexcept>
//==============================================================================
namespace tatooine::gl::glfw{
//==============================================================================
context::context() {
  api::get();
  glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  m_glfw_window = glfwCreateWindow(internal_resolution, internal_resolution, "tatooine context",
                                   nullptr, nullptr);
  if (!m_glfw_window) {
    throw std::runtime_error{"[GLFW] Could not create context."};
  }
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
