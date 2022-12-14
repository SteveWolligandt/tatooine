#ifndef TATOOINE_GL_GLFW_BASE_H
#define TATOOINE_GL_GLFW_BASE_H
//==============================================================================
#include <tatooine/gl/glincludes.h>
//==============================================================================
namespace tatooine::gl::glfw {
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
  auto        make_current() -> void;
  static auto release() -> void;
  auto        get_window_size(int* w, int* h) -> void;
};
//==============================================================================
}  // namespace tatooine::gl::glfw
//==============================================================================
#endif
