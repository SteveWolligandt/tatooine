#ifndef TATOOINE_GL_GLFW_BUTTONS_H
#define TATOOINE_GL_GLFW_BUTTONS_H
//==============================================================================
#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/mouse.h>
//==============================================================================
namespace tatooine::gl::glfw {
//==============================================================================
inline auto convert_button(int const button) {
  switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      return button::left;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      return button::middle;
    case GLFW_MOUSE_BUTTON_RIGHT:
      return button::right;
    default:
      return button::unknown;
  }
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif

