#ifndef TATOOINE_FLOWEXPLORER_NODE_ID_LESS_H
#define TATOOINE_FLOWEXPLORER_NODE_ID_LESS_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct node_id_less {
  bool operator()(const ax::NodeEditor::NodeId& lhs,
                  const ax::NodeEditor::NodeId& rhs) const {
    return lhs.AsPointer() < rhs.AsPointer();
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
