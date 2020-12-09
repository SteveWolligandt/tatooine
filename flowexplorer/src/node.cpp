#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/flowexplorer/ui/pin.h>
//==============================================================================
namespace tatooine::flowexplorer::ui::base {
//==============================================================================
node::node(std::string const& title, flowexplorer::scene& s)
    : m_title{title}, m_scene{&s} {}
//------------------------------------------------------------------------------
node::node(flowexplorer::scene& s) : m_title{""}, m_scene{&s} {}
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
  ed::BeginNode(get_id());
  ImGui::Checkbox("", &m_enabled);
  ImGui::SameLine();
  ImGui::TextUnformatted(title().c_str());
  if (draw_properties()) {
    on_property_changed();
    for (auto& p : m_output_pins) {
      for (auto l : p.links()) {
        (l->input().node().on_property_changed());
      }
    }
  }
  for (auto& input_pin : input_pins()) {
    ed::BeginPin(input_pin.get_id(), ed::PinKind::Input);
    std::string in = "-> " + input_pin.title();
    ImGui::TextUnformatted(in.c_str());
    ed::EndPin();
  }
  for (auto& output_pin : output_pins()) {
    ed::BeginPin(output_pin.get_id(), ed::PinKind::Output);
    std::string out = output_pin.title() + " ->";
    ImGui::TextUnformatted(out.c_str());
    ed::EndPin();
  }
  ed::EndNode();
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
