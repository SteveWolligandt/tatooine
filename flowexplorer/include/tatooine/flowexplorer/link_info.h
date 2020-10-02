#ifndef TATOOINE_FLOWEXPLORER_LINK_INFO_H
#define TATOOINE_FLOWEXPLORER_LINK_INFO_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct link_info {
  ax::NodeEditor::LinkId id;
  ax::NodeEditor::PinId  input_id;
  ax::NodeEditor::PinId  output_id;
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
