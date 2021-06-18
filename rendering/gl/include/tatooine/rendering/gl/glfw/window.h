#ifndef TATOOINE_RENDERING_GL_GLFW_WINDOW_H
#define TATOOINE_RENDERING_GL_GLFW_WINDOW_H
//==============================================================================
#include <yavin/glfw/base.h>
#include <yavin/window_notifier.h>

#include <string>
//==============================================================================
namespace tatooine::rendering::gl::glfw {
//==============================================================================
struct window : base, window_notifier {
  //============================================================================
  window(size_t const width, size_t const height,
         std::string const& title = "yavin window");
  //----------------------------------------------------------------------------
  auto should_close() const -> bool;
  auto swap_buffers() -> void;
};
//==============================================================================
}  // namespace tatooine::rendering::gl::glfw
//==============================================================================
#endif
