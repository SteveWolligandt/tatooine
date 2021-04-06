#include <tatooine/flowexplorer/scene.h>
//
#include <tatooine/flowexplorer/nodes/binary_scalar_operation.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
binary_scalar_operation::binary_scalar_operation(flowexplorer::scene& s)
    : ui::node<binary_scalar_operation>{"Binary Scalar Operation", s, m_value},
      m_s0{insert_input_pin<real_t>("s0")},
      m_s1{insert_input_pin<real_t>("s1")} {}
//------------------------------------------------------------------------------
auto binary_scalar_operation::draw_properties() -> bool {
  ImGui::TextUnformatted(std::to_string(m_value).c_str());
  bool changed = false;
  changed |=
      ImGui::RadioButton("addition",
                         &m_op, (int)op::addition);
  changed |=
      ImGui::RadioButton("subtraction",
                         &m_op, (int)op::subtraction);
  changed |=
      ImGui::RadioButton("mulitplication",
                         &m_op, (int)op::multiplication);
  changed |=
      ImGui::RadioButton("division",
                         &m_op, (int)op::division);
  return changed;
}
//------------------------------------------------------------------------------
auto binary_scalar_operation::on_property_changed() -> void {
  if (m_s0.is_linked() && m_s0.is_linked()) {
    switch (m_op) {
      case (int)op::addition:
        m_value = m_s0.get_linked_as<real_t>() + m_s1.get_linked_as<real_t>();
        break;
      case (int)op::subtraction:
        m_value = m_s0.get_linked_as<real_t>() - m_s1.get_linked_as<real_t>();
        break;
      case (int)op::multiplication:
        m_value = m_s0.get_linked_as<real_t>() * m_s1.get_linked_as<real_t>();
        break;
      case (int)op::division:
        m_value = m_s0.get_linked_as<real_t>() / m_s1.get_linked_as<real_t>();
        break;
    }
    notify_property_changed(false);
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
