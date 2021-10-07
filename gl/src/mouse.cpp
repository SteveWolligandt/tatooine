#include <tatooine/gl/mouse.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
std::string to_string(button b) {
  switch (b) {
    case button::left: return "left";
    case button::right: return "right";
    case button::middle: return "middle";
    case button::wheel_up: return "wheel up";
    case button::wheel_down: return "wheel down";
    case button::wheel_left: return "wheel left";
    case button::wheel_right: return "wheel right";
    default: return "unknown";
  }
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
