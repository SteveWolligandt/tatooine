#include <tatooine/flowexplorer/ui/node_builder.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
node_builder::node_builder()
    : m_cur_node_id{0},
      m_cur_stage{stage::invalid},
      m_has_header{false} {}
//------------------------------------------------------------------------------
auto node_builder::begin(ax::NodeEditor::NodeId id) -> void {
  namespace ed = ax::NodeEditor;
  m_has_header = false;

  ed::PushStyleVar(ed::StyleVar_NodePadding, ImVec4(8, 4, 8, 8));

  ed::BeginNode(id);

  ImGui::PushID(id.AsPointer());
  m_cur_node_id = id;

  set_stage(stage::begin);
}
//------------------------------------------------------------------------------
auto node_builder::end() -> void {
  namespace ed = ax::NodeEditor;
  set_stage(stage::end);

  ed::EndNode();

  if (ImGui::IsItemVisible()) {
    auto alpha = static_cast<int>(255 * ImGui::GetStyle().Alpha);

    //auto drawList = ed::GetNodeBackgroundDrawList(m_cur_node_id);

    //const auto halfBorderWidth = ed::GetStyle().NodeBorderWidth * 0.5f;

    //auto headerColor = IM_COL32(0, 0, 0, alpha) |
    //                   (m_header_color & IM_COL32(255, 255, 255, 0));
    //if ((m_header_max.x > m_header_min.x) &&
    //    (m_header_max.y > m_header_min.y) && m_header_texture_id) {
    //  const auto uv = ImVec2{(m_header_max.x - m_header_min.x} /
    //                             (float)(4.0f * m_header_texture_width),
    //                         (m_header_max.y - m_header_min.y) /
    //                             (float)(4.0f * m_header_texture_height));
    //
    //  drawList->AddImageRounded(
    //      m_header_texture_id,
    //      m_header_min - ImVec2{8 - halfBorderWidth, 4 - halfBorderWidth},
    //      m_header_max + ImVec2{8 - halfBorderWidth, 0}, ImVec2{0.0f, 0.0f}, uv,
    //      headerColor, ed::GetStyle().NodeRounding, 1 | 2);
    //
    //  auto headerSeparatorMin = ImVec2{m_header_min.x, m_header_max.y};
    //  auto headerSeparatorMax = ImVec2{m_header_max.x, m_header_min.y};
    //
    //  if ((headerSeparatorMax.x > headerSeparatorMin.x) &&
    //      (headerSeparatorMax.y > headerSeparatorMin.y)) {
    //    drawList->AddLine(
    //        headerSeparatorMin + ImVec2{-(8 - halfBorderWidth}, -0.5f),
    //        headerSeparatorMax + ImVec2{(8 - halfBorderWidth}, -0.5f),
    //        ImColor(255, 255, 255, 96 * alpha / (3 * 255)), 1.0f);
    //  }
    //}
  }

  m_cur_node_id = 0;

  ImGui::PopID();

  ed::PopStyleVar();

  set_stage(stage::invalid);
}
//------------------------------------------------------------------------------
auto node_builder::header(const ImVec4& color) -> void {
  m_header_color = ImColor(color);
  set_stage(stage::header);
}
//------------------------------------------------------------------------------
auto node_builder::end_header() -> void { set_stage(stage::content); }
//------------------------------------------------------------------------------
auto node_builder::input(ax::NodeEditor::PinId id) -> void {
  if (m_cur_stage == stage::begin) {
    set_stage(stage::content);
  }

  const auto applyPadding = m_cur_stage == stage::input;

  set_stage(stage::input);

  if (applyPadding) {
    ImGui::Spring(0);
  }

  pin(id, ax::NodeEditor::PinKind::Input);

  ImGui::BeginHorizontal(id.AsPointer());
}
//------------------------------------------------------------------------------
auto node_builder::end_input() -> void {
  ImGui::EndHorizontal();
  ax::NodeEditor::EndPin();
}
//------------------------------------------------------------------------------
auto node_builder::middle() -> void {
  if (m_cur_stage == stage::begin) {
    set_stage(stage::content);
  }

  set_stage(stage::middle);
}
//------------------------------------------------------------------------------
auto node_builder::output(ax::NodeEditor::PinId id) -> void {
  if (m_cur_stage == stage::begin) {
    set_stage(stage::content);
  }

  const auto applyPadding = (m_cur_stage == stage::output);

  set_stage(stage::output);

  if (applyPadding) {
    ImGui::Spring(0);
  }

  pin(id, ax::NodeEditor::PinKind::Output);

  ImGui::BeginHorizontal(id.AsPointer());
}
//------------------------------------------------------------------------------
auto node_builder::end_output() -> void {
  ImGui::EndHorizontal();

  ax::NodeEditor::EndPin();
}
//------------------------------------------------------------------------------
auto node_builder::set_stage(stage stage) -> bool {
  namespace ed = ax::NodeEditor;
  if (stage == m_cur_stage) {
    return false;
  }

  auto oldStage = m_cur_stage;
  m_cur_stage   = stage;

  ImVec2 cursor;
  switch (oldStage) {
    case stage::begin:
      break;

    case stage::header:
      ImGui::EndHorizontal();
      //m_header_min = ImGui::GetItemRectMin();
      //m_header_max = ImGui::GetItemRectMax();

      // spacing between header and content
      ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.y * 2.0f);

      break;

    case stage::content:
      break;

    case stage::input:
      ed::PopStyleVar(2);

      ImGui::Spring(1, 0);
      ImGui::EndVertical();

      break;

    case stage::middle:
      ImGui::EndVertical();

      break;

    case stage::output:
      ed::PopStyleVar(2);

      ImGui::Spring(1, 0);
      ImGui::EndVertical();

      break;

    case stage::end:
      break;

    case stage::invalid:
      break;
  }

  switch (stage) {
    case stage::begin:
      ImGui::BeginVertical("node");
      break;

    case stage::header:
      m_has_header = true;

      ImGui::BeginHorizontal("header");
      break;

    case stage::content:
      if (oldStage == stage::begin) {
        ImGui::Spring(0);
      }

      ImGui::BeginHorizontal("content");
      ImGui::Spring(0, 0);
      break;

    case stage::input:
      ImGui::BeginVertical("inputs", ImVec2{0, 0}, 0.0f);

      ed::PushStyleVar(ed::StyleVar_PivotAlignment, ImVec2{0, 0.5f});
      ed::PushStyleVar(ed::StyleVar_PivotSize, ImVec2{0, 0});

      if (!m_has_header) {
        ImGui::Spring(1, 0);
      }
      break;

    case stage::middle:
      ImGui::Spring(1);
      ImGui::BeginVertical("middle", ImVec2{0, 0}, 1.0f);
      break;

    case stage::output:
      if (oldStage == stage::middle || oldStage == stage::input) {
        ImGui::Spring(1);
      } else {
        ImGui::Spring(1, 0);
      }
      ImGui::BeginVertical("outputs", ImVec2{0, 0}, 1.0f);

      ed::PushStyleVar(ed::StyleVar_PivotAlignment, ImVec2{1.0f, 0.5f});
      ed::PushStyleVar(ed::StyleVar_PivotSize, ImVec2{0, 0});

      if (!m_has_header) {
        ImGui::Spring(1, 0);
      }
      break;

    case stage::end:
      if (oldStage == stage::input) {
        ImGui::Spring(1, 0);
      }
      if (oldStage != stage::begin) {
        ImGui::EndHorizontal();
      }
      m_content_min = ImGui::GetItemRectMin();
      m_content_max = ImGui::GetItemRectMax();

      // ImGui::Spring(0);
      ImGui::EndVertical();
      m_node_min = ImGui::GetItemRectMin();
      m_node_max = ImGui::GetItemRectMax();
      break;

    case stage::invalid:
      break;
  }

  return true;
}
//------------------------------------------------------------------------------
auto node_builder::pin(ax::NodeEditor::PinId id, ax::NodeEditor::PinKind kind)
    -> void {
  namespace ed = ax::NodeEditor;
  ed::BeginPin(id, kind);
}
//------------------------------------------------------------------------------
auto node_builder::end_pin() -> void {
  ax::NodeEditor::EndPin();
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
