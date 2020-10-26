#ifndef TATOOINE_FLOWEXPLORER_WINDOW_H
#define TATOOINE_FLOWEXPLORER_WINDOW_H
//==============================================================================
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/flowexplorer/scene.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window : rendering::first_person_window {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  bool                                m_show_nodes_gui = true;
  int                                 mouse_x, mouse_y;
  scene                               m_scene;
  std::unique_ptr<ImGui::FileBrowser> m_file_browser;
  //----------------------------------------------------------------------------
  // ctor
  //----------------------------------------------------------------------------
  window();
  window(std::filesystem::path const& scene_path);
  //----------------------------------------------------------------------------
  ~window() = default;
  //============================================================================
  void on_key_pressed(yavin::key k) override;
  void start();
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
