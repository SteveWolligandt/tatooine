#include <tatooine/gl/imgui_api_backend.h>
#include <tatooine/gl/keyboard.h>

#include <iostream>
#include <memory>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
std::chrono::time_point<std::chrono::system_clock> imgui_api_backend::time =
    std::chrono::system_clock::now();
//==============================================================================
imgui_api_backend::imgui_api_backend() {
  ImGuiIO& io = ImGui::GetIO();
  io.BackendFlags |=
      ImGuiBackendFlags_HasMouseCursors;  // We can honor GetMouseCursor()
                                          // values (optional)
  io.BackendFlags |=
      ImGuiBackendFlags_HasSetMousePos;  // We can honor io.WantSetMousePos
                                         // requests (optional, rarely used)

  io.BackendPlatformName = "imgui_impl_tatooine";

  io.KeyMap[ImGuiKey_Tab]         = static_cast<int>(key::KEY_TAB);
  io.KeyMap[ImGuiKey_LeftArrow]   = static_cast<int>(key::KEY_LEFT);
  io.KeyMap[ImGuiKey_RightArrow]  = static_cast<int>(key::KEY_RIGHT);
  io.KeyMap[ImGuiKey_UpArrow]     = static_cast<int>(key::KEY_UP);
  io.KeyMap[ImGuiKey_DownArrow]   = static_cast<int>(key::KEY_DOWN);
  io.KeyMap[ImGuiKey_PageUp]      = static_cast<int>(key::KEY_PAGE_UP);
  io.KeyMap[ImGuiKey_PageDown]    = static_cast<int>(key::KEY_PAGE_DOWN);
  io.KeyMap[ImGuiKey_Home]        = static_cast<int>(key::KEY_HOME);
  io.KeyMap[ImGuiKey_End]         = static_cast<int>(key::KEY_END);
  io.KeyMap[ImGuiKey_Insert]      = static_cast<int>(key::KEY_INSERT);
  io.KeyMap[ImGuiKey_Delete]      = static_cast<int>(key::KEY_DELETE);
  io.KeyMap[ImGuiKey_Backspace]   = static_cast<int>(key::KEY_BACKSPACE);
  io.KeyMap[ImGuiKey_Space]       = static_cast<int>(key::KEY_SPACE);
  io.KeyMap[ImGuiKey_Enter]       = static_cast<int>(key::KEY_ENTER);
  io.KeyMap[ImGuiKey_Escape]      = static_cast<int>(key::KEY_ESCAPE);
  io.KeyMap[ImGuiKey_KeyPadEnter] = static_cast<int>(key::KEY_KP_ENTER);
  io.KeyMap[ImGuiKey_A]           = static_cast<int>(key::KEY_A);
  io.KeyMap[ImGuiKey_C]           = static_cast<int>(key::KEY_C);
  io.KeyMap[ImGuiKey_V]           = static_cast<int>(key::KEY_V);
  io.KeyMap[ImGuiKey_X]           = static_cast<int>(key::KEY_X);
  io.KeyMap[ImGuiKey_Y]           = static_cast<int>(key::KEY_Y);
  io.KeyMap[ImGuiKey_Z]           = static_cast<int>(key::KEY_Z);
}
//------------------------------------------------------------------------------
imgui_api_backend::~imgui_api_backend() {}
//------------------------------------------------------------------------------
void imgui_api_backend::on_key_pressed(key k) {
  ImGuiIO&   io     = ImGui::GetIO();
  auto const k_id   = static_cast<uint8_t>(k);
  io.KeysDown[k_id] = true;
  if (k == key::KEY_CTRL_L || k == key::KEY_CTRL_R) {
    io.KeyCtrl = true;
  } else if (k == key::KEY_SHIFT_L || k == key::KEY_SHIFT_R) {
    io.KeyShift = true;
  } else if (k == key::KEY_ALT_L || k == key::KEY_ALT_R) {
    io.KeyAlt = true;
  } else if (k_id >= static_cast<std::uint8_t>(key::KEY_0) &&
             k_id <= static_cast<std::uint8_t>(key::KEY_9)) {
    io.AddInputCharacter(static_cast<std::uint8_t>('0') +
                         (k_id - static_cast<std::uint8_t>(key::KEY_0)));
  } else if (k == key::KEY_SPACE) {
    io.AddInputCharacter((unsigned int)(' '));
  } else if (k_id >= static_cast<std::uint8_t>(key::KEY_A) &&
             k_id <= static_cast<std::uint8_t>(key::KEY_Z)) {
    if (io.KeyShift) {
      io.AddInputCharacter(static_cast<uint8_t>('A') +
                           (k_id - static_cast<std::uint8_t>(key::KEY_A)));
    } else {
      io.AddInputCharacter(static_cast<uint8_t>('a') +
                           (k_id - static_cast<std::uint8_t>(key::KEY_A)));
    }
  } else if (k == key::KEY_DECIMALPOINT) {
    io.AddInputCharacter(static_cast<unsigned int>('.'));
  } else if (k == key::KEY_MINUS) {
    io.AddInputCharacter(static_cast<unsigned int>('-'));
  }
}
//------------------------------------------------------------------------------
auto imgui_api_backend::on_key_released(key const k) -> void {
  auto& io    = ImGui::GetIO();
  io.KeysDown[static_cast<std::uint8_t>(k)] = false;
  if (k == key::KEY_CTRL_L || k == key::KEY_CTRL_R) {
    io.KeyCtrl = false;
  }
  if (k == key::KEY_SHIFT_L || k == key::KEY_SHIFT_R) {
    io.KeyShift = false;
  }
  if (k == key::KEY_ALT_L || k == key::KEY_ALT_R) {
    io.KeyAlt = false;
  }
}
//------------------------------------------------------------------------------
void imgui_api_backend::on_button_pressed(button b) {
  ImGuiIO& io = ImGui::GetIO();
  switch (b) {
    case button::left:
      io.MouseDown[0] = true;
      break;
    case button::right:
      io.MouseDown[1] = true;
      break;
    case button::middle:
      io.MouseDown[2] = true;
      break;
    case button::unknown:
    default:
      break;
  }
}
//------------------------------------------------------------------------------
void imgui_api_backend::on_button_released(button b) {
  ImGuiIO& io = ImGui::GetIO();
  switch (b) {
    case button::left:
      io.MouseDown[0] = false;
      break;
    case button::right:
      io.MouseDown[1] = false;
      break;
    case button::middle:
      io.MouseDown[2] = false;
      break;
    case button::unknown:
    default:
      break;
  }
}
//------------------------------------------------------------------------------
imgui_api_backend& imgui_api_backend::instance() {
  static auto inst = std::make_unique<imgui_api_backend>();
  return *inst;
}
//------------------------------------------------------------------------------
void imgui_api_backend::on_resize(int w, int h) {
  ImGuiIO& io    = ImGui::GetIO();
  io.DisplaySize = ImVec2((float)w, (float)h);
}
//------------------------------------------------------------------------------
void imgui_api_backend::on_cursor_moved(double x, double y) {
  ImGuiIO& io = ImGui::GetIO();
  io.MousePos = ImVec2((float)x, (float)y);
}
//------------------------------------------------------------------------------
void imgui_api_backend::new_frame() {
  // Setup time step
  auto&      io           = ImGui::GetIO();
  auto const current_time = std::chrono::system_clock::now();
  auto       delta_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time)
          .count();
  if (delta_time_ms <= 0) {
    delta_time_ms = 1;
  }
  io.DeltaTime = static_cast<float>(delta_time_ms) / 1000.0f;

  time = current_time;
  // Start the frame
  ImGui::NewFrame();
}
//------------------------------------------------------------------------------
void imgui_api_backend::on_mouse_wheel(int dir) {
  ImGuiIO& io = ImGui::GetIO();
  if (dir > 0) {
    io.MouseWheel += 1.0f;
  } else if (dir < 0) {
    io.MouseWheel -= 1.0f;
  }
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
