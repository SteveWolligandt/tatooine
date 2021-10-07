#include <tatooine/gl/imgui_api_backend.h>
#include <tatooine/gl/keyboard.h>
#include <memory>
#include <iostream>
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

  io.KeyMap[ImGuiKey_Tab]         = KEY_TAB;
  io.KeyMap[ImGuiKey_LeftArrow]   = KEY_LEFT;
  io.KeyMap[ImGuiKey_RightArrow]  = KEY_RIGHT;
  io.KeyMap[ImGuiKey_UpArrow]     = KEY_UP;
  io.KeyMap[ImGuiKey_DownArrow]   = KEY_DOWN;
  io.KeyMap[ImGuiKey_PageUp]      = KEY_PAGE_UP;
  io.KeyMap[ImGuiKey_PageDown]    = KEY_PAGE_DOWN;
  io.KeyMap[ImGuiKey_Home]        = KEY_HOME;
  io.KeyMap[ImGuiKey_End]         = KEY_END;
  io.KeyMap[ImGuiKey_Insert]      = KEY_INSERT;
  io.KeyMap[ImGuiKey_Delete]      = KEY_DELETE;
  io.KeyMap[ImGuiKey_Backspace]   = KEY_BACKSPACE;
  io.KeyMap[ImGuiKey_Space]       = KEY_SPACE;
  io.KeyMap[ImGuiKey_Enter]       = KEY_ENTER;
  io.KeyMap[ImGuiKey_Escape]      = KEY_ESCAPE;
  io.KeyMap[ImGuiKey_KeyPadEnter] = KEY_KP_ENTER;
  io.KeyMap[ImGuiKey_A]           = KEY_A;
  io.KeyMap[ImGuiKey_C]           = KEY_C;
  io.KeyMap[ImGuiKey_V]           = KEY_V;
  io.KeyMap[ImGuiKey_X]           = KEY_X;
  io.KeyMap[ImGuiKey_Y]           = KEY_Y;
  io.KeyMap[ImGuiKey_Z]           = KEY_Z;
}
//------------------------------------------------------------------------------
imgui_api_backend::~imgui_api_backend() {}
//------------------------------------------------------------------------------
void imgui_api_backend::on_key_pressed(key k) {
  ImGuiIO& io = ImGui::GetIO();
  unsigned int k_id = static_cast<unsigned int>(k);
  io.KeysDown[k]    = true;
  if (k == KEY_CTRL_L || k == KEY_CTRL_R) {
    io.KeyCtrl = true;
  } else if (k == KEY_SHIFT_L || k == KEY_SHIFT_R) {
    io.KeyShift = true;
  } else if (k == KEY_ALT_L || k == KEY_ALT_R) {
    io.KeyAlt = true;
  } else if (k_id >= KEY_0 && k_id <= KEY_9) {
    io.AddInputCharacter((unsigned int)('0' + (unsigned int)(k - KEY_0)));
  } else if (k_id == KEY_SPACE) {
    io.AddInputCharacter((unsigned int)(' '));
  } else if (k_id >= KEY_A && k_id <= KEY_Z) {
    if (io.KeyShift) {
      io.AddInputCharacter((unsigned int)('A' + (unsigned int)(k - KEY_A)));
    } else {
      io.AddInputCharacter((unsigned int)('a' + (unsigned int)(k - KEY_A)));
    }
  } else if (k_id == KEY_DECIMALPOINT) {
    io.AddInputCharacter((unsigned int)('.'));
  } else if (k_id == KEY_MINUS) {
    io.AddInputCharacter((unsigned int)('-'));
  }
}
//------------------------------------------------------------------------------
void imgui_api_backend::on_key_released(key k) {
  ImGuiIO& io    = ImGui::GetIO();
  io.KeysDown[k] = false;
  if (k == KEY_CTRL_L || k == KEY_CTRL_R) {
    io.KeyCtrl = false;
  }
  if (k == KEY_SHIFT_L || k == KEY_SHIFT_R) {
    io.KeyShift = false;
  }
  if (k == KEY_ALT_L || k == KEY_ALT_R) {
    io.KeyAlt = false;
  }
}
//------------------------------------------------------------------------------
void imgui_api_backend::on_button_pressed(button b) {
  ImGuiIO& io = ImGui::GetIO();
  switch(b){
    case button::left: io.MouseDown[0] = true; break;
    case button::right: io.MouseDown[1] = true; break;
    case button::middle: io.MouseDown[2] = true; break;
    case button::unknown:
    default: break;
  }
}
//------------------------------------------------------------------------------
void imgui_api_backend::on_button_released(button b) {
  ImGuiIO& io = ImGui::GetIO();
  switch (b) {
    case button::left: io.MouseDown[0] = false; break;
    case button::right: io.MouseDown[1] = false; break;
    case button::middle: io.MouseDown[2] = false; break;
    case button::unknown:
    default: break;
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
  ImGuiIO& io            = ImGui::GetIO();
  auto      current_time  = std::chrono::system_clock::now();
  int       delta_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(current_time - time)
          .count();
  if (delta_time_ms <= 0) delta_time_ms = 1;
  io.DeltaTime = delta_time_ms / 1000.0;

  time = current_time;
  // Start the frame
  ImGui::NewFrame();
}
//------------------------------------------------------------------------------
void imgui_api_backend::on_mouse_wheel(int dir) {
  ImGuiIO& io = ImGui::GetIO();
  if (dir > 0) {
    io.MouseWheel += 1.0;
  } else if (dir < 0) {
    io.MouseWheel -= 1.0;
  }
}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
