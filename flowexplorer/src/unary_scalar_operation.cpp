#include <tatooine/flowexplorer/scene.h>
//
#include <tatooine/flowexplorer/nodes/unary_scalar_operation.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
unary_scalar_operation::unary_scalar_operation(flowexplorer::scene& s)
    : ui::node<unary_scalar_operation>{"Unary Scalar Operation", s, m_value},
      m_input{insert_input_pin<real_type>("s0")} {}
//------------------------------------------------------------------------------
auto unary_scalar_operation::draw_properties() -> bool {
  ImGui::TextUnformatted(std::to_string(m_value).c_str());
  bool changed = false;
  changed |=
      ImGui::RadioButton("sine",
                         &m_op, (int)op::sin);
  changed |=
      ImGui::RadioButton("cosine",
                         &m_op, (int)op:: cos);
  return changed;
}
//------------------------------------------------------------------------------
auto unary_scalar_operation::on_property_changed() -> void {
  if (m_input.is_linked() && m_input.is_linked()) {
    switch (m_op) {
      case (int)op::sin:
        m_value = std::sin(m_input.get_linked_as<real_type>());
        break;
      case (int)op::cos:
        m_value = std::cos(m_input.get_linked_as<real_type>());
        break;
    }
    notify_property_changed(false);
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
