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
      return button::BUTTON_LEFT;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      return button::BUTTON_MIDDLE;
    case GLFW_MOUSE_BUTTON_RIGHT:
      return button::BUTTON_RIGHT;
    default:
      return button::BUTTON_UNKNOWN;
  }
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif

