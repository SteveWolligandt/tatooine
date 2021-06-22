#ifndef TATOOINE_GL_MOUSE_H
#define TATOOINE_GL_MOUSE_H
//==============================================================================
#include <string>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
enum button {
  BUTTON_LEFT,
  BUTTON_RIGHT,
  BUTTON_MIDDLE,
  BUTTON_WHEEL_UP,
  BUTTON_WHEEL_DOWN,
  BUTTON_WHEEL_LEFT,
  BUTTON_WHEEL_RIGHT,
  BUTTON_UNKNOWN
};

auto to_string(button b) -> std::string;

struct button_listener {
  virtual void on_button_pressed(button /*b*/) {}
  virtual void on_button_released(button /*b*/) {}
  virtual void on_wheel_up() {}
  virtual void on_wheel_down() {}
  virtual void on_wheel_left() {}
  virtual void on_wheel_right() {}
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
