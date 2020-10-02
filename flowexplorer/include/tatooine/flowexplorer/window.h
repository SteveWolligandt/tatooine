#ifndef TATOOINE_FLOWEXPLORER_WINDOW_H
#define TATOOINE_FLOWEXPLORER_WINDOW_H
//==============================================================================
#include <tatooine/rendering/first_person_window.h>
#include <tatooine/flowexplorer/renderable.h>
#include <tatooine/flowexplorer/link_info.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window : rendering::first_person_window {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  bool                                     m_show_nodes_gui;
  std::vector<std::unique_ptr<renderable>> m_renderables;
  std::vector<std::unique_ptr<ui::node>>   m_nodes;

  int mouse_x, mouse_y;

  ax::NodeEditor::EditorContext* m_node_editor_context = nullptr;
  ImVector<link_info> m_links;  // List of live links. It is dynamic unless you
                                // want to create read-only view over nodes.
  int m_next_link = 100;

  //----------------------------------------------------------------------------
  // ctor
  //----------------------------------------------------------------------------
  window();
  //----------------------------------------------------------------------------
  ~window(); 
  //============================================================================
  void on_key_pressed(yavin::key k) override;
  void start();
  //----------------------------------------------------------------------------
  auto find_node(ax::NodeEditor::NodeId id) -> renderable*;
  //----------------------------------------------------------------------------
  auto find_pin(ax::NodeEditor::PinId id) -> ui::pin*;
  //----------------------------------------------------------------------------
  void node_creators();
  //----------------------------------------------------------------------------
  void draw_nodes();
  //----------------------------------------------------------------------------
  void draw_links();
  //----------------------------------------------------------------------------
  void create_link();
  //----------------------------------------------------------------------------
  void remove_link();
  //----------------------------------------------------------------------------
  void node_editor();
  //----------------------------------------------------------------------------
  void render_ui();
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
