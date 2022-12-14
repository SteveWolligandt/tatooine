#ifndef TATOOINE_FLOWEXPLORER_WINDOW_H
#define TATOOINE_FLOWEXPLORER_WINDOW_H
//==============================================================================
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window : rendering::first_person_window {
  using this_type   = window;
  using parent_type = rendering::first_person_window;
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  float                               m_ui_scale_factor = 1.0f;
  bool                                m_show_nodes_gui = true;
  double                              m_mouse_x, m_mouse_y;
  bool                                m_left_button_down = false;
  scene                               m_scene;
  std::unique_ptr<ImGui::FileBrowser> m_file_browser;
  ui::base::node*                     m_path_notifier = nullptr;
  ImFont*                             m_font_regular = nullptr;
  ImFont*                             m_font_header1 = nullptr;
  ImFont*                             m_font_header2 = nullptr;
  ImFont*                             m_font_bold = nullptr;
  gl::tex2rgba32f          m_aabb2d_icon_tex;
  gl::tex2rgba32f                  m_aabb3d_icon_tex;
  //============================================================================
  auto ui_scale_factor() const { return m_ui_scale_factor; }
  //----------------------------------------------------------------------------
  auto aabb2d_icon_tex() const -> auto const& { return m_aabb2d_icon_tex; }
  auto aabb3d_icon_tex() const -> auto const& { return m_aabb3d_icon_tex; }

  //----------------------------------------------------------------------------
  // ctor
  //----------------------------------------------------------------------------
  window();
  window(filesystem::path const& scene_path);
  //----------------------------------------------------------------------------
  ~window();
  //============================================================================
  void on_key_pressed(gl::key k) override;
  void on_button_pressed(gl::button b) override;
  void on_button_released(gl::button b) override;
  void on_cursor_moved(double /*x*/, double /*y*/) override;
  void start();
  //=============================================================================
  auto push_regular_font() -> void { ImGui::PushFont(m_font_regular); }
  auto push_header1_font() -> void { ImGui::PushFont(m_font_header1); }
  auto push_header2_font() -> void { ImGui::PushFont(m_font_header2); }
  auto push_bold_font() -> void { ImGui::PushFont(m_font_bold); }
  auto pop_font() -> void { ImGui::PopFont(); }
  //----------------------------------------------------------------------------
  auto close_file_explorer() {
    m_file_browser->ClearSelected();
    m_file_browser->Close();
    m_path_notifier = nullptr;
  }
  //----------------------------------------------------------------------------
 private:
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
                          std::vector<std::string> const& extensions) {
    open_file_explorer(title);
    m_file_browser->SetTypeFilters(extensions);
  }

  //----------------------------------------------------------------------------
 public:
  auto open_file_explorer(ui::base::node& n) {
    m_file_browser  = std::make_unique<ImGui::FileBrowser>(0);
    m_path_notifier = &n;
    m_file_browser->Open();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto open_file_explorer(std::string const& title, ui::base::node& n) {
    open_file_explorer(n);
    m_file_browser->SetTitle(title);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto open_file_explorer(std::string const&              title,
                          std::vector<std::string> const& extensions,
                          ui::base::node& n) {
    open_file_explorer(title, n);
    m_file_browser->SetTypeFilters(extensions);
  }
  //----------------------------------------------------------------------------
  auto open_file_explorer_write(ui::base::node& n) {
    m_file_browser  = std::make_unique<ImGui::FileBrowser>(ImGuiFileBrowserFlags_EnterNewFilename);
    m_path_notifier = &n;
    m_file_browser->Open();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto open_file_explorer_write(std::string const& title, ui::base::node& n) {
    open_file_explorer_write(n);
    m_file_browser->SetTitle(title);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto open_file_explorer_write(std::string const&              title,
                          std::vector<std::string> const& extensions,
                          ui::base::node& n) {
    open_file_explorer_write(title, n);
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
