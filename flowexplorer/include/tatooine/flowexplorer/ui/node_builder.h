#ifndef TATOOINE_FLOWEXPLORER_UI_NODE_BUILDER_H
#define TATOOINE_FLOWEXPLORER_UI_NODE_BUILDER_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/gl/imgui.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
struct node_builder {
 private:
  enum class stage {
    invalid,
    begin,
    header,
    content,
    input,
    output,
    middle,
    end
  };
  //============================================================================
  ax::NodeEditor::NodeId m_cur_node_id;
  stage                  m_cur_stage;
  ImU32                  m_header_color;
  ImVec2                 m_node_min;
  ImVec2                 m_node_max;
  ImVec2                 m_header_min;
  ImVec2                 m_header_max;
  ImVec2                 m_content_min;
  ImVec2                 m_content_max;
  bool                   m_has_header;

 public:
  node_builder();

  auto begin(ax::NodeEditor::NodeId id) -> void;
  auto end() -> void;

  auto header(const ImVec4& color = ImVec4(1, 1, 1, 1)) -> void;
  auto end_header() -> void;

  auto input(ax::NodeEditor::PinId id) -> void;
  auto end_input() -> void;

  auto middle() -> void;

  auto output(ax::NodeEditor::PinId id) -> void;
  auto end_output() -> void;

 private:
  auto set_stage(stage) -> bool;

  auto pin(ax::NodeEditor::PinId id, ax::NodeEditor::PinKind kind) -> void;
  auto end_pin() -> void;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#endif
