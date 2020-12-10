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
  bool                                m_show_nodes_gui = true;
  int                                 m_mouse_x, m_mouse_y;
  bool                                m_left_button_down = false;
  scene                               m_scene;
  std::unique_ptr<ImGui::FileBrowser> m_file_browser;
  ImFont*                             m_font = nullptr;

  auto font() -> auto& { return *m_font; }
  auto font() const -> auto const& { return *m_font; }
  //----------------------------------------------------------------------------
  // ctor
  //----------------------------------------------------------------------------
  window();
  window(std::filesystem::path const& scene_path);
  //----------------------------------------------------------------------------
  ~window() = default;
  //============================================================================
  void on_key_pressed(yavin::key k) override;
  void on_button_pressed(yavin::button b) override;
  void on_button_released(yavin::button b) override;
  void on_mouse_motion(int /*x*/, int /*y*/) override;
  void start();
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
