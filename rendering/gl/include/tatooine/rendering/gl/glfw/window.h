#ifndef YAVIN_GLFW_WINDOW_H
#define YAVIN_GLFW_WINDOW_H
//==============================================================================
#include <yavin/glfw/base.h>
#include <yavin/window_notifier.h>

#include <string>
//==============================================================================
namespace yavin::glfw {
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
}  // namespace yavin::glfw
//==============================================================================
#endif
