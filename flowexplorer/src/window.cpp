#include <tatooine/flowexplorer/window.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
window::window() : m_scene{camera_controller(), this} { start(); }
//------------------------------------------------------------------------------
window::window(std::filesystem::path const& path)
    : m_scene{camera_controller(), this, path} {
  start();
}
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
    m_file_browser = std::make_unique<ImGui::FileBrowser>(0);
    m_file_browser->SetTitle("Load File");
    m_file_browser->SetTypeFilters({".toml", ".scene", ".vtk"});
    m_file_browser->Open();
  } else if (k == yavin::KEY_ESCAPE) {
    if (m_file_browser) {
      if (m_file_browser->IsOpened()) {
        m_file_browser->Close();
      }
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
    std::cerr << ray.origin() << " -> " << ray.direction() << '\n';
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
  parent_t::on_mouse_motion(x,y);
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
        for (auto& output_pin : r->output_pins()) {
          if (output_pin.is_connected()) {
            output_pin.link().input().node().on_property_changed();
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
      m_scene.draw_node_editor(0, 0, this->width() / 3, this->height(),
                               m_show_nodes_gui);
    }
    if (m_file_browser) {
      m_file_browser->Display();
      if (m_file_browser->HasSelected()) {
        m_scene.open_file(m_file_browser->GetSelected());
        m_file_browser->ClearSelected();
        m_file_browser->Close();
      }
    }
  });
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
