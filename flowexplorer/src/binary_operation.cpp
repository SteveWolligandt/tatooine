#include <tatooine/flowexplorer/scene.h>
//
#include <tatooine/flowexplorer/nodes/binary_operation.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
binary_operation::binary_operation(flowexplorer::scene& s)
    : ui::node<binary_operation>{"Binary Operation", s},
      m_input0{insert_input_pin<real_type,
                                polymorphic::vectorfield<real_type, 2>,
                                polymorphic::vectorfield<real_type, 3>,
                                polymorphic::vectorfield<real_type, 4>,
                                polymorphic::matrixfield<real_type, 2>,
                                polymorphic::matrixfield<real_type, 3>,
                                polymorphic::matrixfield<real_type, 4>>("")},
      m_input1{insert_input_pin<real_type,
                                polymorphic::vectorfield<real_type, 2>,
                                polymorphic::vectorfield<real_type, 3>,
                                polymorphic::vectorfield<real_type, 4>,
                                polymorphic::matrixfield<real_type, 2>,
                                polymorphic::matrixfield<real_type, 3>,
                                polymorphic::matrixfield<real_type, 4>>("")},
      m_scalar_pin_out{insert_output_pin<real_type>(
          "", *reinterpret_cast<real_type*>(&std::get<0>(m_output_data)))},
      m_dot_field2_pin_out{
          insert_output_pin<polymorphic::scalarfield<real_type, 2>>(
              "", *reinterpret_cast<polymorphic::scalarfield<real_type, 2>*>(
                      &std::get<0>(m_output_data)))},
      m_dot_field3_pin_out{
          insert_output_pin<polymorphic::scalarfield<real_type, 3>>(
              "", *reinterpret_cast<polymorphic::scalarfield<real_type, 3>*>(
                      &std::get<0>(m_output_data)))},
      m_mat_vec_mult_field2_pin_out{
          insert_output_pin<polymorphic::vectorfield<real_type, 2>>(
              "", *reinterpret_cast<polymorphic::vectorfield<real_type, 2>*>(
                      &std::get<0>(m_output_data)))},
      m_mat_vec_mult_field3_pin_out{
          insert_output_pin<polymorphic::vectorfield<real_type, 3>>(
              "", *reinterpret_cast<polymorphic::vectorfield<real_type, 3>*>(
                      &std::get<0>(m_output_data)))} {
  deactivate_output_pins();
}
//------------------------------------------------------------------------------
auto binary_operation::draw_properties() -> bool {
  bool changed = false;
  if (m_input0.is_linked() && m_input1.is_linked()) {
    if (m_input0.linked_type() == typeid(real_type) &&
        m_input1.linked_type() == typeid(real_type)) {
      changed |= ImGui::RadioButton("addition", &m_operation,
                                    (int)operation_t::addition);
      changed |= ImGui::RadioButton("subtraction", &m_operation,
                                    (int)operation_t::subtraction);
      changed |= ImGui::RadioButton("mulitplication", &m_operation,
                                    (int)operation_t::multiplication);
      changed |= ImGui::RadioButton("division", &m_operation,
                                    (int)operation_t::division);
      ImGui::TextUnformatted(
          std::to_string(std::get<real_type>(m_output_data)).c_str());
    } else if (m_input0.linked_type() ==
                   typeid(polymorphic::vectorfield<real_type, 2>) &&
               m_input1.linked_type() ==
                   typeid(polymorphic::vectorfield<real_type, 2>)) {
      if (ImGui::RadioButton("addition", &m_operation,
                             (int)operation_t::addition)) {
        changed = true;
      }
      if (ImGui::RadioButton("subtraction", &m_operation,
                             (int)operation_t::subtraction)) {
        changed = true;
      }
      if (ImGui::RadioButton("mulitplication", &m_operation,
                             (int)operation_t::multiplication)) {
        changed = true;
      }
      if (ImGui::RadioButton("division", &m_operation,
                             (int)operation_t::division)) {
        changed = true;
      }
      if (ImGui::RadioButton("dot", &m_operation, (int)operation_t::dot)) {
        m_output_data = dot_field2_t{
            &m_input0.get_linked_as<polymorphic::vectorfield<real_type, 2>>(),
            &m_input1.get_linked_as<polymorphic::vectorfield<real_type, 2>>(), dot};
        m_dot_field2_pin_out.activate();
        changed = true;
      }
    }
  }
  return changed;
}
//------------------------------------------------------------------------------
auto binary_operation::on_property_changed() -> void {
  if (m_input0.is_linked() && m_input0.is_linked()) {
    if (m_input0.linked_type() == typeid(real_type) &&
        m_input1.linked_type() == typeid(real_type)) {
      auto& scalar_out = std::get<real_type>(m_output_data);
      switch (m_operation) {
        case (int)operation_t::addition:
          scalar_out = m_input0.get_linked_as<real_type>() +
                       m_input1.get_linked_as<real_type>();
          break;
        case (int)operation_t::subtraction:
          scalar_out = m_input0.get_linked_as<real_type>() -
                       m_input1.get_linked_as<real_type>();
          break;
        case (int)operation_t::multiplication:
          scalar_out = m_input0.get_linked_as<real_type>() *
                       m_input1.get_linked_as<real_type>();
          break;
        case (int)operation_t::division:
          scalar_out = m_input0.get_linked_as<real_type>() /
                       m_input1.get_linked_as<real_type>();
          break;
      }
    }
    notify_property_changed(false);
  }
}
//------------------------------------------------------------------------------
auto binary_operation::on_pin_disconnected(ui::input_pin&) -> void {
  deactivate_output_pins();
  m_output_data = std::monostate{};
}
//------------------------------------------------------------------------------
auto binary_operation::on_pin_connected(ui::input_pin&, ui::output_pin&)
    -> void {
  deactivate_output_pins();
  if (m_input0.is_linked() && m_input1.is_linked()) {
    if (m_input0.linked_type() == typeid(real_type) &&
        m_input1.linked_type() == typeid(real_type)) {
      m_scalar_pin_out.activate();
      on_property_changed();
    } else if (m_input0.linked_type() ==
                   typeid(polymorphic::vectorfield<real_type, 2>) &&
               m_input1.linked_type() ==
                   typeid(polymorphic::vectorfield<real_type, 2>)) {
      on_property_changed();
    } else if (m_input0.linked_type() ==
                   typeid(polymorphic::vectorfield<real_type, 3>) &&
               m_input1.linked_type() ==
                   typeid(polymorphic::vectorfield<real_type, 3>)) {
      m_dot_field3_pin_out.activate();
      auto& dot_field3 = std::get<dot_field3_t>(m_output_data);
      dot_field3.set_v0(
          &m_input0.get_linked_as<polymorphic::vectorfield<real_type, 3>>());
      dot_field3.set_v1(
          &m_input1.get_linked_as<polymorphic::vectorfield<real_type, 3>>());
      on_property_changed();
    } else if (m_input0.linked_type() ==
                   typeid(polymorphic::matrixfield<real_type, 2>) &&
               m_input1.linked_type() ==
                   typeid(polymorphic::vectorfield<real_type, 2>)) {
      m_mat_vec_mult_field2_pin_out.activate();
      auto& mat_vec_mult_field2 =
          std::get<mat_vec_mult_field2_t>(m_output_data);
      mat_vec_mult_field2.set_v0(
          &m_input0.get_linked_as<polymorphic::matrixfield<real_type, 2>>());
      mat_vec_mult_field2.set_v1(
          &m_input1.get_linked_as<polymorphic::vectorfield<real_type, 2>>());
      on_property_changed();
    } else if (m_input0.linked_type() ==
                   typeid(polymorphic::matrixfield<real_type, 3>) &&
               m_input1.linked_type() ==
                   typeid(polymorphic::vectorfield<real_type, 3>)) {
      m_mat_vec_mult_field3_pin_out.activate();
      auto& mat_vec_mult_field3 =
          std::get<mat_vec_mult_field3_t>(m_output_data);
      mat_vec_mult_field3.set_v0(
          &m_input0.get_linked_as<polymorphic::matrixfield<real_type, 3>>());
      mat_vec_mult_field3.set_v1(
          &m_input1.get_linked_as<polymorphic::vectorfield<real_type, 3>>());
      on_property_changed();
    }
  }
}
//==============================================================================
auto binary_operation::deactivate_output_pins() -> void {
  m_scalar_pin_out.deactivate();
  m_dot_field2_pin_out.deactivate();
  m_dot_field3_pin_out.deactivate();
  m_mat_vec_mult_field2_pin_out.deactivate();
  m_mat_vec_mult_field3_pin_out.deactivate();
}
//==============================================================================
  }  // namespace tatooine::flowexplorer::nodes
  //==============================================================================
