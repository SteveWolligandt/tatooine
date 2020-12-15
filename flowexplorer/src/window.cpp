#include <tatooine/flowexplorer/directories.h>
#include <tatooine/flowexplorer/window.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
window::window()
    : m_scene{camera_controller(), this},
      m_aabb2d_icon_tex{icons_directory() / "aabb2d.png"},
      m_aabb3d_icon_tex{icons_directory() / "aabb3d.png"} {
  if (display().x11_display().screen_resolution().first > 2000) {
    m_ui_scale_factor = 2.0f;
  }
  m_font_regular = ImGui::GetIO().Fonts->AddFontFromFileTTF(
      (fonts_directory() / "Roboto-Regular.ttf").c_str(),
      15.0f * m_ui_scale_factor);
  m_font_bold = ImGui::GetIO().Fonts->AddFontFromFileTTF(
      (fonts_directory() / "Roboto-Bold.ttf").c_str(),
      20.0f * m_ui_scale_factor);
  imgui_render_backend().create_fonts_texture();

  if (std::filesystem::exists("scene.toml")) {
    m_scene.read("scene.toml");
  }
  start();
}
//------------------------------------------------------------------------------
window::window(std::filesystem::path const& path)
    : m_scene{camera_controller(), this, path},
      m_aabb2d_icon_tex{icons_directory() / "aabb2d.png"},
      m_aabb3d_icon_tex{icons_directory() / "aabb3d.png"} {
  if (display().x11_display().screen_resolution().first > 2000) {
    m_ui_scale_factor = 2.0f;
  }
  m_font_regular = ImGui::GetIO().Fonts->AddFontFromFileTTF(
      (fonts_directory() / "Roboto-Regular.ttf").c_str(),
      15.0f * m_ui_scale_factor);
  m_font_bold = ImGui::GetIO().Fonts->AddFontFromFileTTF(
      (fonts_directory() / "Roboto-Bold.ttf").c_str(),
      20.0f * m_ui_scale_factor);
  imgui_render_backend().create_fonts_texture();

  if (std::filesystem::exists("scene.toml")) {
    m_scene.read("scene.toml");
  }
  start();
}
//------------------------------------------------------------------------------
window::~window() { m_scene.write("scene.toml"); }
//==============================================================================
void window::on_key_pressed(yavin::key k) {
  parent_t::on_key_pressed(k);
  if (k == yavin::KEY_F1) {
    m_show_nodes_gui = !m_show_nodes_gui;
  } else if (k == yavin::KEY_F2) {
    camera_controller().use_perspective_camera();
    camera_controller().use_fps_controller();
  } else if (k == yavin::KEY_F3) {
    camera_controller().use_orthographic_camera();
    camera_controller().use_orthographic_controller();
  } else if (k == yavin::KEY_F4) {
    camera_controller().look_at({0, 0, 0}, {0, 0, -1});
  } else if (k == yavin::KEY_F5) {
    m_scene.write("scene.toml");
  } else if (k == yavin::KEY_F6) {
    if (!file_explorer_is_opened()) {
      m_picking_file = true;
      open_file_explorer("Load File", {".toml", ".scene", ".vtk"});
    }
  } else if (k == yavin::KEY_ESCAPE) {
    if (file_explorer_is_opened()) {
      close_file_explorer();
      m_picking_file = false;
    }
  }
}
//------------------------------------------------------------------------------
void window::on_button_pressed(yavin::button b) {
  parent_t::on_button_pressed(b);
  if (ImGui::GetIO().WantCaptureMouse) {
    return;
  }
  if (b == yavin::BUTTON_LEFT) {
    m_left_button_down = true;
    auto const ray =
        m_scene.camera_controller().ray(m_mouse_x, height() - m_mouse_y - 1);
    for (auto& r : m_scene.renderables()) {
      if (r->check_intersection(ray)) {
        r->on_mouse_clicked();
      }
    }
  }
}
//------------------------------------------------------------------------------
void window::on_button_released(yavin::button b) {
  parent_t::on_button_released(b);
  if (ImGui::GetIO().WantCaptureMouse) {
    return;
  }
  if (b == yavin::BUTTON_LEFT) {
    m_left_button_down = true;
    for (auto& r : m_scene.renderables()) {
      if (r->is_picked()) {
        r->on_mouse_released();
      }
    }
  }
}
//------------------------------------------------------------------------------
void window::on_mouse_motion(int x, int y) {
  parent_t::on_mouse_motion(x, y);
  if (ImGui::GetIO().WantCaptureMouse) {
    return;
  }
  if (m_left_button_down) {
    auto offset_x = x - m_mouse_x;
    auto offset_y = y - m_mouse_y;
    if (offset_x == 0 && offset_y == 0) {
      return;
    }
    for (auto& r : m_scene.renderables()) {
      if (r->is_picked() && r->on_mouse_drag(offset_x, offset_y)) {
        if (r->has_self_pin()) {
          for (auto l : r->self_pin().links()) {
            l->input().node().on_property_changed();
          }
        }
        for (auto& output_pin : r->output_pins()) {
          for (auto l : output_pin.links()) {
            l->input().node().on_property_changed();
          }
        }
      }
    }
  }
  m_mouse_x = x;
  m_mouse_y = y;
}
//------------------------------------------------------------------------------
void window::start() {
  render_loop([&](const auto& dt) {
    m_scene.render(dt);
    if (m_show_nodes_gui) {
      m_scene.draw_node_editor(0, this->height() * 2 / 3, this->width(),
                               this->height() / 3, m_show_nodes_gui);
      // m_scene.draw_node_editor(0, 0, this->width(), this->height(),
      //                         m_show_nodes_gui);
    }
    if (m_file_browser) {
      m_file_browser->Display();
      if (m_picking_file) {
        if (m_file_browser->HasSelected()) {
          m_scene.open_file(m_file_browser->GetSelected());
          close_file_explorer();
        }
      }
    }
  });
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
