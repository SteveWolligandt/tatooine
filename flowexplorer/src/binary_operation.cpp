#include <tatooine/flowexplorer/scene.h>
//
#include <tatooine/flowexplorer/nodes/binary_operation.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
binary_operation::binary_operation(flowexplorer::scene& s)
    : ui::node<binary_operation>{"Binary Operation", s},
      m_scalar_value{0.0},
      m_matrix_vector_multiplication_field{nullptr, nullptr, mult},
      m_input0{
          insert_input_pin<real_t, polymorphic::matrixfield<real_t, 3>>("")},
      m_input1{
          insert_input_pin<real_t, polymorphic::vectorfield<real_t, 3>>("")},
      m_scalar_pin_out{insert_output_pin<real_t>("", m_scalar_value)},
      m_vectorfield3_pin_out{
          insert_output_pin<polymorphic::vectorfield<real_t, 3>>(
              "", m_matrix_vector_multiplication_field)} {
  m_scalar_pin_out.deactivate();
  m_vectorfield3_pin_out.deactivate();
}
//------------------------------------------------------------------------------
auto binary_operation::draw_properties() -> bool {
  bool changed = false;
  if (m_input0.is_linked() && m_input1.is_linked()) {
    if (m_input0.linked_type() == typeid(real_t) &&
        m_input1.linked_type() == typeid(real_t)) {
      changed |= ImGui::RadioButton("addition", &m_operation,
                                    (int)operation_t::addition);
      changed |= ImGui::RadioButton("subtraction", &m_operation,
                                    (int)operation_t::subtraction);
      changed |= ImGui::RadioButton("mulitplication", &m_operation,
                                    (int)operation_t::multiplication);
      changed |= ImGui::RadioButton("division", &m_operation,
                                    (int)operation_t::division);
      ImGui::TextUnformatted(std::to_string(m_scalar_value).c_str());
    }
  }
  return changed;
}
//------------------------------------------------------------------------------
auto binary_operation::on_property_changed() -> void {
  if (m_input0.is_linked() && m_input0.is_linked()) {
    if (m_input0.linked_type() == typeid(real_t) &&
        m_input1.linked_type() == typeid(real_t)) {
      switch (m_operation) {
        case (int)operation_t::addition:
          m_scalar_value = m_input0.get_linked_as<real_t>() +
                           m_input1.get_linked_as<real_t>();
          break;
        case (int)operation_t::subtraction:
          m_scalar_value = m_input0.get_linked_as<real_t>() -
                           m_input1.get_linked_as<real_t>();
          break;
        case (int)operation_t::multiplication:
          m_scalar_value = m_input0.get_linked_as<real_t>() *
                           m_input1.get_linked_as<real_t>();
          break;
        case (int)operation_t::division:
          m_scalar_value = m_input0.get_linked_as<real_t>() /
                           m_input1.get_linked_as<real_t>();
          break;
      }
    }
    notify_property_changed(false);
  }
}
//------------------------------------------------------------------------------
auto binary_operation::on_pin_disconnected(ui::input_pin&) -> void {
  m_scalar_pin_out.deactivate();
  m_vectorfield3_pin_out.deactivate();
}
//------------------------------------------------------------------------------
auto binary_operation::on_pin_connected(ui::input_pin&, ui::output_pin&)
    -> void {
  if (m_input0.is_linked() && m_input1.is_linked()) {
    if (m_input0.linked_type() == typeid(real_t) &&
        m_input1.linked_type() == typeid(real_t)) {
      m_scalar_pin_out.activate();
      m_vectorfield3_pin_out.deactivate();
      on_property_changed();
    } else if (m_input0.linked_type() ==
                   typeid(polymorphic::matrixfield<real_t, 3>) &&
               m_input1.linked_type() == typeid(polymorphic::vectorfield<real_t, 3>)) {
      m_scalar_pin_out.deactivate();
      m_vectorfield3_pin_out.activate();
      m_matrix_vector_multiplication_field.set_v0(
          &m_input0.get_linked_as<polymorphic::matrixfield<real_t, 3>>());
      m_matrix_vector_multiplication_field.set_v1(
          &m_input1.get_linked_as<polymorphic::vectorfield<real_t, 3>>());
      on_property_changed();
    }
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
