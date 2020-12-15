#ifndef TATOOINE_FLOWEXPLORER_WINDOW_H
#define TATOOINE_FLOWEXPLORER_WINDOW_H
//==============================================================================
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/flowexplorer/scene.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window : rendering::first_person_window {
  using this_t   = window;
  using parent_t = rendering::first_person_window;
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  float                               m_ui_scale_factor = 1.0f;
  bool                                m_show_nodes_gui = true;
  int                                 m_mouse_x, m_mouse_y;
  bool                                m_left_button_down = false;
  scene                               m_scene;
  std::unique_ptr<ImGui::FileBrowser> m_file_browser;
  ImFont*                             m_font_regular = nullptr;
  ImFont*                             m_font_bold = nullptr;
  yavin::tex2rgba32f                  m_aabb2d_icon_tex;
  yavin::tex2rgba32f                  m_aabb3d_icon_tex;
  bool                                m_picking_file = false;
  //============================================================================
  auto ui_scale_factor() const { return m_ui_scale_factor; }
  //----------------------------------------------------------------------------
  auto aabb2d_icon_tex() const -> auto const& { return m_aabb2d_icon_tex; }
  auto aabb3d_icon_tex() const -> auto const& { return m_aabb3d_icon_tex; }

  //----------------------------------------------------------------------------
  // ctor
  //----------------------------------------------------------------------------
  window();
  window(std::filesystem::path const& scene_path);
  //----------------------------------------------------------------------------
  ~window();
  //============================================================================
  void on_key_pressed(yavin::key k) override;
  void on_button_pressed(yavin::button b) override;
  void on_button_released(yavin::button b) override;
  void on_mouse_motion(int /*x*/, int /*y*/) override;
  void start();
  //=============================================================================
  auto push_regular_font() -> void { ImGui::PushFont(m_font_regular); }
  auto push_bold_font() -> void { ImGui::PushFont(m_font_bold); }
  auto pop_font() -> void { ImGui::PopFont(); }
  //----------------------------------------------------------------------------
  auto close_file_explorer() {
    m_file_browser->ClearSelected();
    m_file_browser->Close();
  }
  //----------------------------------------------------------------------------
  auto open_file_explorer() {
    m_file_browser = std::make_unique<ImGui::FileBrowser>(0);
    m_file_browser->Open();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto open_file_explorer(std::string const& title) {
    open_file_explorer();
    m_file_browser->SetTitle(title);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto open_file_explorer(std::string const&              title,
                          std::vector<char const*> const& extensions) {
    open_file_explorer(title);
    m_file_browser->SetTypeFilters(extensions);
  }
  //----------------------------------------------------------------------------
  auto file_explorer_is_opened() const {
    return m_file_browser != nullptr && m_file_browser->IsOpened();
  }
  //----------------------------------------------------------------------------
  auto file_explorer() const -> auto const& { return *m_file_browser; }
  auto file_explorer() -> auto& { return *m_file_browser; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
