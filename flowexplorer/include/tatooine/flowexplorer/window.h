#ifndef TATOOINE_FLOWEXPLORER_WINDOW_H
#define TATOOINE_FLOWEXPLORER_WINDOW_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/boundingbox.h>
#include <tatooine/demangling.h>
#include <tatooine/flowexplorer/nodes/abcflow.h>
#include <tatooine/flowexplorer/nodes/boundingbox.h>
#include <tatooine/flowexplorer/nodes/doublegyre.h>
#include <tatooine/flowexplorer/nodes/duffing_oscillator.h>
#include <tatooine/flowexplorer/nodes/lic.h>
#include <tatooine/flowexplorer/nodes/random_pathlines.h>
#include <tatooine/flowexplorer/nodes/rayleigh_benard_convection.h>
#include <tatooine/flowexplorer/nodes/spacetime_vectorfield.h>
#include <tatooine/interpolation.h>
#include <tatooine/rendering/first_person_window.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct window : rendering::first_person_window {
  struct NodeIdLess {
    bool operator()(const ax::NodeEditor::NodeId& lhs,
                    const ax::NodeEditor::NodeId& rhs) const {
      return lhs.AsPointer() < rhs.AsPointer();
    }
  };
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
  bool                                     m_show_nodes_gui;
  std::vector<std::unique_ptr<renderable>> m_renderables;
  std::vector<std::unique_ptr<ui::node>>   m_nodes;

  int mouse_x, mouse_y;

  struct LinkInfo {
    ax::NodeEditor::LinkId id;
    ax::NodeEditor::PinId  input_id;
    ax::NodeEditor::PinId  output_id;
  };
  ax::NodeEditor::EditorContext* m_node_editor_context = nullptr;
  ImVector<LinkInfo> m_links;  // List of live links. It is dynamic unless you
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
