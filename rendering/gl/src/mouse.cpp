#include <yavin/mouse.h>
//==============================================================================
namespace yavin {
//==============================================================================
std::string to_string(button b) {
  switch (b) {
    case button::BUTTON_LEFT: return "left";
    case button::BUTTON_RIGHT: return "right";
    case button::BUTTON_MIDDLE: return "middle";
    case button::BUTTON_WHEEL_UP: return "wheel up";
    case button::BUTTON_WHEEL_DOWN: return "wheel down";
    case button::BUTTON_WHEEL_LEFT: return "wheel left";
    case button::BUTTON_WHEEL_RIGHT: return "wheel right";
    default: return "unknown";
  }
}
//==============================================================================
}  // namespace yavin
//==============================================================================
