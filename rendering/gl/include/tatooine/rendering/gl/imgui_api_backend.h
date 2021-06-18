#ifndef TATOOINE_RENDERING_GL_IMGUI_API_BACKEND_H
#define TATOOINE_RENDERING_GL_IMGUI_API_BACKEND_H
//==============================================================================
#include <yavin/glincludes.h>
#include <yavin/imgui_includes.h>
#include <chrono>
#include "window_listener.h"
//==============================================================================
namespace tatooine::rendering::gl {
//==============================================================================
struct imgui_api_backend  {
  static imgui_api_backend& instance();
  static std::chrono::time_point<std::chrono::system_clock> time;
  //----------------------------------------------------------------------------
  imgui_api_backend();
  virtual ~imgui_api_backend();
  //----------------------------------------------------------------------------
  void on_key_pressed(key k) ;
  void on_key_released(key k) ;
  void on_button_pressed(button b) ;
  void on_button_released(button b) ;
  void on_cursor_moved(double x, double y) ;
  void on_resize(int width, int height) ;
  void on_mouse_wheel(int dir);
  void new_frame();
};
//==============================================================================
}  // namespace tatooine::rendering::gl
//==============================================================================
#endif
