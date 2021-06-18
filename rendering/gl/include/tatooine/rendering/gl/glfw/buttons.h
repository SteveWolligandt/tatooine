#ifndef TATOOINE_RENDERING_GL_GLFW_BUTTONS_H
#define TATOOINE_RENDERING_GL_GLFW_BUTTONS_H
//==============================================================================
#include <yavin/glincludes.h>
#include <yavin/mouse.h>
//==============================================================================
namespace tatooine::rendering::gl::glfw {
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
}  // namespace tatooine::rendering::gl
//==============================================================================
#endif

