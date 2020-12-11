#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/flowexplorer/ui/node_builder.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/ui/draw_icon.h>
//==============================================================================
namespace tatooine::flowexplorer::ui::base {
//==============================================================================
node::node(flowexplorer::scene& s) : m_title{""}, m_scene{&s} {}
//------------------------------------------------------------------------------
node::node(flowexplorer::scene& s, std::type_info const& type) : node{s} {
  m_self_pin = std::make_unique<output_pin>(*this, type, "");
}
//------------------------------------------------------------------------------
node::node(std::string const& title, flowexplorer::scene& s)
    : m_title{title}, m_scene{&s} {}
//------------------------------------------------------------------------------
node::node(std::string const& title, flowexplorer::scene& s,
           std::type_info const& type)
    : node{title, s} {
  m_self_pin = std::make_unique<output_pin>(*this, type, "");
}
//------------------------------------------------------------------------------
auto node::node_position() const -> ImVec2 {
  ImVec2 pos;
  m_scene->do_in_context(
      [&] { pos = ax::NodeEditor::GetNodePosition(get_id()); });
  return pos;
}
//------------------------------------------------------------------------------
auto node::draw_node() -> void {
  namespace ed = ax::NodeEditor;
  node_builder builder;
  builder.begin(get_id());

  ImGui::Dummy(ImVec2(10, 0));
  builder.header();
  ImGui::Checkbox("", &m_enabled);
  ImGui::Spring(0);
  scene().window().push_bold_font();
  ImGui::TextUnformatted(title().c_str());
  scene().window().pop_font();
  ImGui::Spring(1);
  //ImGui::Dummy(ImVec2(0, 28));

  if (m_self_pin != nullptr) {
    ImGui::BeginVertical("delegates", ImVec2(0, 0));
    ImGui::Spring(1, 0);
    auto alpha = ImGui::GetStyle().Alpha;
    // if (newLinkPin && !CanCreateLink(newLinkPin, &m_self_pin) &&
    //    &m_self_pin != newLinkPin)
    //  alpha = alpha * (48.0f / 255.0f);

    ed::BeginPin(m_self_pin->get_id(), ed::PinKind::Output);
    ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
    ed::PinPivotSize(ImVec2(0, 0));
    ImGui::BeginHorizontal(m_self_pin->get_id().AsPointer());
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
    if (!m_self_pin->title().empty()) {
      ImGui::TextUnformatted(m_self_pin->title().c_str());
      ImGui::Spring(0);
    }
    // DrawPinIcon(m_self_pin, IsPinLinked(m_self_pin->get_id()), (int)(alpha *
    // 255));
    icon(ImVec2(50, 50), icon_type::flow, m_self_pin->is_connected());
    //std::string out = "-> ";
    //ImGui::TextUnformatted(out.c_str());
    ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.x / 2);
    ImGui::EndHorizontal();
    ImGui::PopStyleVar();
    ed::EndPin();

    // DrawItemRect(ImColor(255, 0, 0));
    ImGui::Spring(1, 0);
    ImGui::EndVertical();
    ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.x / 2);
  } else {
    ImGui::Spring(0);
  }
  builder.end_header();

  for (auto& input : input_pins()) {
     auto alpha = ImGui::GetStyle().Alpha;
    // if (newLinkPin && !CanCreateLink(newLinkPin, &input) &&
    //    &input != newLinkPin)
    //  alpha = alpha * (48.0f / 255.0f);

    builder.input(input.get_id());
     ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
    icon(ImVec2(50, 50), icon_type::flow, input.is_connected());
    ImGui::Spring(0);
    if (!input.title().empty()) {
      ImGui::TextUnformatted(input.title().c_str());
      ImGui::Spring(0);
    }
    ImGui::PopStyleVar();
    builder.end_input();
  }
  // ImGui::Spring(0);
  builder.middle();
  ImGui::PushItemWidth(400);
  if (draw_properties()) {
    on_property_changed();
    for (auto& p : m_output_pins) {
      for (auto l : p.links()) {
        (l->input().node().on_property_changed());
      }
    }
  }
  ImGui::PopItemWidth();
  for (auto& output : output_pins()) {
     auto alpha = ImGui::GetStyle().Alpha;
    // if (newLinkPin && !CanCreateLink(newLinkPin, &output) &&
    //    &output != newLinkPin)
    //  alpha = alpha * (48.0f / 255.0f);

     ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
    builder.output(output.get_id());
    if (!output.title().empty()) {
      ImGui::Spring(0);
      ImGui::TextUnformatted(output.title().c_str());
    }
    ImGui::Spring(0);
    // DrawPinIcon(output, IsPinLinked(output.get_id()), (int)(alpha * 255));
    icon(ImVec2(50, 50), icon_type::flow, output.is_connected());
     ImGui::PopStyleVar();
    builder.end_output();
  }
  builder.end();
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui::base
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
auto insert_registered_element(scene& s, std::string_view const& name)
    -> ui::base::node* {
  iterate_registered_factories(factory) {
    if (auto ptr = factory->f(s, name); ptr) {
      return ptr;
    }
  }
  return nullptr;
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
