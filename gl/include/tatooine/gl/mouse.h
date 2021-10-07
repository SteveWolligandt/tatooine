#ifndef TATOOINE_GL_MOUSE_H
#define TATOOINE_GL_MOUSE_H
//==============================================================================
#include <string>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
enum class button {
  left,
  right,
  middle,
  wheel_up,
  wheel_down,
  wheel_left,
  wheel_right,
  unknown
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
