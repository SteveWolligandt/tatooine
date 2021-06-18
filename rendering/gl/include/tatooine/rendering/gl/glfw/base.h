#ifndef TATOOINE_RENDERING_GL_GLFW_BASE_H
#define TATOOINE_RENDERING_GL_GLFW_BASE_H
//==============================================================================
#include <tatooine/rendering/gl/glincludes.h>
//==============================================================================
namespace tatooine::rendering::gl::glfw {
//==============================================================================
struct base {
 protected:
  GLFWwindow* m_glfw_window;

 public:
  base();
  ~base();
  //----------------------------------------------------------------------------
  auto get() -> GLFWwindow*;
  auto get() const -> GLFWwindow const*;
  //----------------------------------------------------------------------------
  auto make_current() -> void;
  static auto release() -> void;
};
//==============================================================================
}  // namespace tatooine::rendering::gl::glfw
//==============================================================================
#endif
