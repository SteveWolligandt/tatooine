#ifndef TATOOINE_RENDERING_GL_WINDOW_LISTENER_H
#define TATOOINE_RENDERING_GL_WINDOW_LISTENER_H
//==============================================================================
#include "keyboard.h"
#include <iostream>
#include "mouse.h"
//==============================================================================
namespace tatooine::rendering::gl {
//==============================================================================
struct window_listener : keyboard_listener, button_listener {
  virtual void on_cursor_moved(double /*x*/, double /*y*/) {}
  virtual void on_resize(int /*width*/, int /*height*/) {}
  virtual void on_close() { std::cerr << "close\n"; }
};
//==============================================================================
}  // namespace tatooine::rendering::gl
//==============================================================================
#endif
