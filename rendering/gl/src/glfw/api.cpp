#include <yavin/glfw/api.h>
#include <yavin/glfw/window.h>
#include <yavin/glfw/buttons.h>
#include <yavin/glfw/keys.h>
//==============================================================================
namespace yavin::glfw {
//==============================================================================
api::api() {
  glfwSetErrorCallback(api::on_error);
  if (!glfwInit()) {
    throw std::runtime_error{"[GLFW] Could not initialize."};
  }
}
//----------------------------------------------------------------------------
api::~api() { glfwTerminate(); }
//============================================================================
auto api::get() -> api const& {
  static api singleton{};
  return singleton;
}
//----------------------------------------------------------------------------
// GLFW callbacks
//----------------------------------------------------------------------------
auto api::on_error(int /*error*/, char const* description) -> void {
  std::cerr << "[GLFW] error:\n  " << description << '\n';
}
//----------------------------------------------------------------------------
auto api::on_close(GLFWwindow* w) -> void {
  if (auto ptr = glfwGetWindowUserPointer(w); ptr != nullptr) {
    static_cast<window*>(ptr)->notify_close();
  }
}
//----------------------------------------------------------------------------
auto api::on_resize(GLFWwindow* w, int width, int height) -> void {
  if (auto ptr = glfwGetWindowUserPointer(w); ptr != nullptr) {
    static_cast<window*>(ptr)->notify_resize(width, height);
  }
}
//----------------------------------------------------------------------------
auto api::on_cursor_moved(GLFWwindow* w, double xpos, double ypos) -> void {
  if (auto ptr = glfwGetWindowUserPointer(w); ptr != nullptr) {
    static_cast<window*>(ptr)->notify_cursor_moved(xpos, ypos);
  }
}
//----------------------------------------------------------------------------
auto api::on_button(GLFWwindow* w, int button, int action, int /*mods*/)
    -> void {
  if (auto ptr = glfwGetWindowUserPointer(w); ptr != nullptr) {
    if (action == GLFW_PRESS) {
      static_cast<window*>(ptr)->notify_button_pressed(convert_button(button));
    } else if (action == GLFW_RELEASE) {
      static_cast<window*>(ptr)->notify_button_released(convert_button(button));
    }
  }
}
//----------------------------------------------------------------------------
auto api::on_mouse_wheel(GLFWwindow* w, double xoffset, double yoffset)
    -> void {
  if (auto ptr = glfwGetWindowUserPointer(w); ptr != nullptr) {
    if (yoffset > 0) {
      static_cast<window*>(ptr)->notify_wheel_up();
    } else if (yoffset < 0) {
      static_cast<window*>(ptr)->notify_wheel_down();
    } else if (xoffset > 0) {
      static_cast<window*>(ptr)->notify_wheel_left();
    } else if (xoffset < 0) {
      static_cast<window*>(ptr)->notify_wheel_right();
    }
  }
}
//----------------------------------------------------------------------------
auto api::on_key(GLFWwindow* w, int key, int /*scancode*/, int action,
                 int /*mods*/) -> void {
  if (auto ptr = glfwGetWindowUserPointer(w); ptr != nullptr) {
    if (action == GLFW_PRESS) {
      static_cast<window*>(ptr)->notify_key_pressed(convert_key(key));
    } else if (action == GLFW_RELEASE) {
      static_cast<window*>(ptr)->notify_key_released(convert_key(key));
    }
  }
}
//==============================================================================
}  // namespace yavin::glfw
//==============================================================================
