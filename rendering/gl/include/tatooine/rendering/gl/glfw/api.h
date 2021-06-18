#ifndef TATOOINE_RENDERING_GL_GLFW_API_H
#define TATOOINE_RENDERING_GL_GLFW_API_H
//==============================================================================
#include <tatooine/rendering/gl/glincludes.h>
#include <iostream>
#include <stdexcept>
//==============================================================================
namespace tatooine::rendering::gl::glfw {
//==============================================================================
struct api {
  static auto get() -> api const&;
  //----------------------------------------------------------------------------
  // GLFW callbacks
  //----------------------------------------------------------------------------
  static auto on_error(int /*error*/, char const* description) -> void;
  static auto on_close(GLFWwindow*) -> void;
  static auto on_resize(GLFWwindow* w, int width, int height) -> void;
  static auto on_cursor_moved(GLFWwindow* w, double xpos, double ypos) -> void;
  static auto on_button(GLFWwindow*, int button, int action, int mods) -> void;
  static auto on_mouse_wheel(GLFWwindow*, double xoffset, double yoffset)
      -> void;
  static auto on_key(GLFWwindow*, int key, int scancode, int action, int mods)
      -> void;

 private:
  api();

 public:
  ~api();
  };
//==============================================================================
}  // namespace tatooine::rendering::gl::glfw
//==============================================================================
#endif
