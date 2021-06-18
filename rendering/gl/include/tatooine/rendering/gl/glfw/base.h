#ifndef YAVIN_GLFW_BASE_H
#define YAVIN_GLFW_BASE_H
//==============================================================================
#include <yavin/glincludes.h>
//==============================================================================
namespace yavin::glfw {
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
}  // namespace yavin::glfw
//==============================================================================
#endif