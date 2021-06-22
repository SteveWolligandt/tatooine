#ifndef TATOOINE_GL_WINDOW_LISTENER_H
#define TATOOINE_GL_WINDOW_LISTENER_H
//==============================================================================
#include "keyboard.h"
#include <iostream>
#include "mouse.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
struct window_listener : keyboard_listener, button_listener {
  virtual void on_cursor_moved(double /*x*/, double /*y*/) {}
  virtual void on_resize(int /*width*/, int /*height*/) {}
  virtual void on_close() { std::cerr << "close\n"; }
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
