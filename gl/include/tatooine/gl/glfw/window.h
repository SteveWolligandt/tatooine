#ifndef TATOOINE_GL_GLFW_WINDOW_H
#define TATOOINE_GL_GLFW_WINDOW_H
//==============================================================================
#include <tatooine/gl/glfw/base.h>
#include <tatooine/gl/window_notifier.h>

#include <string>
//==============================================================================
namespace tatooine::gl::glfw {
//==============================================================================
struct window : base, window_notifier {
  //============================================================================
  window(size_t const width, size_t const height,
         std::string const& title = "tatooine");
  //----------------------------------------------------------------------------
  auto should_close() const -> bool;
  auto swap_buffers() -> void;
};
//==============================================================================
}  // namespace tatooine::gl::glfw
//==============================================================================
#endif
