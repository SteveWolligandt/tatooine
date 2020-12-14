#include <tatooine/flowexplorer/scene.h>
#include <tatooine/field.h>
#include <tatooine/flowexplorer/nodes/axis_aligned_bounding_box.h>
#include <tatooine/flowexplorer/nodes/position.h>
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
auto icon_color(std::type_info const& type, float const alpha) -> ImVec4 {
  if (type == typeid(parent::vectorfield<double, 2>) ||
      type == typeid(parent::vectorfield<double, 3>)) {
    return {1, 0.5, 0.5, alpha};
  } else if (type == typeid(nodes::axis_aligned_bounding_box<2>) ||
             type == typeid(nodes::axis_aligned_bounding_box<3>)) {
    return {0.5, 1, 0.5, alpha};
  } else if (type == typeid(nodes::position<2>) ||
             type == typeid(nodes::position<3>)) {
    return {0.5, 0.5, 1, alpha};
  }
  return {1, 1, 1, alpha};
}
//------------------------------------------------------------------------------
auto icon_color(ui::output_pin const& pin, float const alpha) -> ImVec4 {
  if (pin.type() == typeid(parent::vectorfield<double, 2>) ||
      pin.type() == typeid(parent::vectorfield<double, 3>)) {
    return icon_color(typeid(parent::vectorfield<double, 2>), alpha);
  } else if (pin.type() == typeid(nodes::axis_aligned_bounding_box<2>) ||
             pin.type() == typeid(nodes::axis_aligned_bounding_box<3>)) {
    return icon_color(typeid(nodes::axis_aligned_bounding_box<2>), alpha);
  } else if (pin.type() == typeid(nodes::position<2>) ||
             pin.type() == typeid(nodes::position<3>)) {
    return icon_color(typeid(nodes::position<2>), alpha);
  }
  return {1, 1, 1, alpha};
}
//------------------------------------------------------------------------------
auto icon_color(ui::input_pin const& pin, float const alpha) -> ImVec4 {
  if (std::any_of(begin(pin.types()), end(pin.types()), [](auto t) {
        return *t == typeid(parent::vectorfield<double, 2>) ||
               *t == typeid(parent::vectorfield<double, 3>);
      })) {
    return icon_color(typeid(parent::vectorfield<double, 2>), alpha);
  } else if (std::any_of(begin(pin.types()), end(pin.types()), [](auto t) {
               return *t == typeid(nodes::axis_aligned_bounding_box<2>) ||
                      *t == typeid(nodes::axis_aligned_bounding_box<3>);
             })) {
    return icon_color(typeid(nodes::axis_aligned_bounding_box<2>), alpha);
  } else if (std::any_of(begin(pin.types()), end(pin.types()), [](auto t) {
               return *t == typeid(nodes::position<2>) ||
                      *t == typeid(nodes::position<3>);
             })) {
    return icon_color(typeid(nodes::position<2>), alpha);
  }
  return {1, 1, 1, alpha};
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

  auto alpha = ImGui::GetStyle().Alpha;
  if (m_self_pin != nullptr && scene().new_link() &&
      !scene().can_create_new_link(*m_self_pin)) {
    alpha = alpha * 48.0f / 255.0f;
  }
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);

  scene().window().push_bold_font();
  ImGui::TextUnformatted(title().c_str());
  scene().window().pop_font();
  ImGui::PopStyleVar();

  ImGui::Spring(1);

  if (m_self_pin != nullptr) {
    ImGui::BeginVertical("delegates", ImVec2(0, 0));
    ImGui::Spring(1, 0);
    auto alpha = ImGui::GetStyle().Alpha;
    if (scene().new_link() && !scene().can_create_new_link(*m_self_pin)) {
      alpha = alpha * 48.0f / 255.0f;
    }

    ed::BeginPin(m_self_pin->get_id(), ed::PinKind::Output);
    ed::PinPivotAlignment(ImVec2(1.0f, 0.5f));
    ed::PinPivotSize(ImVec2(0, 0));
    ImGui::BeginHorizontal(m_self_pin->get_id().AsPointer());
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
    if (!m_self_pin->title().empty()) {
      ImGui::TextUnformatted(m_self_pin->title().c_str());
      ImGui::Spring(0);
    }
    icon(ImVec2(25 * scene().window().ui_scale_factor(),
                25 * scene().window().ui_scale_factor()),
         icon_type::flow, m_self_pin->is_connected(),
         icon_color(*m_self_pin, alpha));
    ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.x / 2);
    ImGui::EndHorizontal();
    ImGui::PopStyleVar();
    ed::EndPin();

    ImGui::Spring(1, 0);
    ImGui::EndVertical();
    ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.x / 2);
  } else {
    ImGui::Spring(0);
  }
  builder.end_header();

  for (auto& input : input_pins()) {
    auto alpha = ImGui::GetStyle().Alpha;
    if (scene().new_link() && !scene().can_create_new_link(input)) {
      alpha = alpha * 48.0f / 255.0f;
    }

    builder.input(input.get_id());
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
    icon(ImVec2(25 * scene().window().ui_scale_factor(),
                25 * scene().window().ui_scale_factor()),
        icon_type::flow, input.is_connected(), icon_color(input, alpha));
    ImGui::Spring(0);
    if (!input.title().empty()) {
      ImGui::TextUnformatted(input.title().c_str());
      ImGui::Spring(0);
    }
    ImGui::PopStyleVar();
    builder.end_input();
  }
  // ImGui::Spring(0);
  {
    auto alpha = ImGui::GetStyle().Alpha;
    if (scene().new_link()) {
      alpha = alpha * 48.0f / 255.0f;
    }
    builder.middle();
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
    ImGui::PushItemWidth(200 * scene().window().ui_scale_factor());
    if (draw_properties()) {
      on_property_changed();
      if (has_self_pin()) {
        for (auto l : m_self_pin->links()) {
          l->input().node().on_property_changed();
        }
      }
      for (auto& p : m_output_pins) {
        for (auto l : p.links()) {
          l->input().node().on_property_changed();
        }
      }
    }
    ImGui::PopItemWidth();
    ImGui::PopStyleVar();
  }

  for (auto& output : output_pins()) {
    auto alpha = ImGui::GetStyle().Alpha;
    if (scene().new_link() && !scene().can_create_new_link(output)) {
      alpha = alpha * 48.0f / 255.0f;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
    builder.output(output.get_id());
    if (!output.title().empty()) {
      ImGui::Spring(0);
      ImGui::TextUnformatted(output.title().c_str());
    }
    ImGui::Spring(0);
    icon(ImVec2(25 * scene().window().ui_scale_factor(),
                25 * scene().window().ui_scale_factor()),
         icon_type::flow, output.is_connected(), icon_color(output, alpha));
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
