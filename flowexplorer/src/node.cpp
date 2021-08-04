#include <tatooine/flowexplorer/scene.h>
#include <tatooine/field.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/flowexplorer/ui/node_builder.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/ui/draw_icon.h>
//==============================================================================
namespace tatooine::flowexplorer::ui::base {
//==============================================================================
node::node(flowexplorer::scene& s) : m_title{""}, m_scene{&s} {}
//------------------------------------------------------------------------------
node::node(std::string const& title, flowexplorer::scene& s)
    : m_title{title}, m_scene{&s} {}
//------------------------------------------------------------------------------
auto node::notify_property_changed(bool const notify_self) -> void {
  if (notify_self) {
    on_property_changed();
  }
  if (has_self_pin()) {
    for (auto l : m_self_pin->links()) {
      l->input().node().on_property_changed();
    }
  }
  for (auto& p : m_output_pins) {
    for (auto l : p->links()) {
      l->input().node().on_property_changed();
    }
  }
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
  size_t const icon_size = 20 * scene().window().ui_scale_factor();
  namespace ed = ax::NodeEditor;
  node_builder builder;
  builder.begin(get_id());

  ImGui::Dummy(ImVec2(10, 0));
  builder.header();
  ImGui::Checkbox("", &is_active());
  ImGui::SameLine();

  auto alpha = ImGui::GetStyle().Alpha;
  if (m_self_pin != nullptr && scene().new_link() &&
      !scene().can_create_new_link(*m_self_pin)) {
    alpha = alpha * 48.0f / 255.0f;
  }

  // draw editable title
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
  scene().window().push_bold_font();
  ImGui::PushItemWidth(200);
  ImGui::TextUnformatted(title().c_str());
  ImGui::PopItemWidth();
  scene().window().pop_font();
  ImGui::PopStyleVar();


  if (m_self_pin != nullptr) {
    ImGui::SameLine();
    auto alpha = ImGui::GetStyle().Alpha;
    if (scene().new_link() && !scene().can_create_new_link(*m_self_pin)) {
      alpha = alpha * 48.0f / 255.0f;
    }

    ed::BeginPin(m_self_pin->get_id(), ed::PinKind::Output);
    ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
    ed::PinPivotSize(ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
    if (!m_self_pin->title().empty()) {
      ImGui::TextUnformatted(m_self_pin->title().c_str());
    }
    m_self_pin->draw(icon_size, alpha);
    ImGui::PopStyleVar();
    ed::EndPin();

  }
  builder.end_header();

  ImGui::Columns(2);
  for (auto& input : input_pins()) {
    auto alpha = ImGui::GetStyle().Alpha;
    if (scene().new_link() && !scene().can_create_new_link(*input)) {
      alpha = alpha * 48.0f / 255.0f;
    }

    builder.input(input->get_id());
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
    input->draw(icon_size, alpha);
    if (!input->title().empty()) {
      ImGui::SameLine();
      ImGui::TextUnformatted(input->title().c_str());
    }
    ImGui::PopStyleVar();
    builder.end_input();
  }

  ImGui::NextColumn();
  for (auto& output : output_pins()) {
    if (output->is_active()) {
      auto alpha = ImGui::GetStyle().Alpha;
      if (scene().new_link() && !scene().can_create_new_link(*output)) {
        alpha = alpha * 48.0f / 255.0f;
      }

      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
      builder.output(output->get_id());
      if (!output->title().empty()) {
        ImGui::TextUnformatted(output->title().c_str());
        ImGui::SameLine();
      }
      output->draw(icon_size, alpha);
      ImGui::PopStyleVar();
      builder.end_output();
    }
  }
  ImGui::Columns(1);
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
