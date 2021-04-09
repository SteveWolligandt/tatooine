#ifndef TATOOINE_FLOWEXPLORER_NODES_BINARY_OPERATION_H
#define TATOOINE_FLOWEXPLORER_NODES_BINARY_OPERATION_H
//==============================================================================
#include <tatooine/field_operations.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct binary_operation : ui::node<binary_operation> {
  enum class operation_t : int {
    addition,
    subtraction,
    multiplication,
    division
  };
  static constexpr auto mult = [](const auto& lhs, const auto& rhs) {
    return lhs * rhs;
  };
  using matrix_vector_field_multiplication_t =
      binary_operation_field<parent::field<real_t, 3, 3, 3>*,
                             parent::field<real_t, 3, 3>*, decltype(mult),
                             real_t, 3, 3>;

  real_t                               m_scalar_value;
  matrix_vector_field_multiplication_t m_matrix_vector_multiplication_field;
  int                                  m_operation = 0;
  ui::input_pin&                       m_input0;
  ui::input_pin&                       m_input1;
  ui::output_pin&                      m_scalar_pin_out;
  ui::output_pin&                      m_vectorfield3_pin_out;

  binary_operation(flowexplorer::scene& s);
  virtual ~binary_operation() = default;
  auto draw_properties() -> bool override;
  auto on_property_changed() -> void override;
  auto on_pin_connected(ui::input_pin&, ui::output_pin&) -> void override;
  auto on_pin_disconnected(ui::input_pin&) -> void override;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::binary_operation,
    TATOOINE_REFLECTION_INSERT_METHOD(scalar_operation, m_operation));
#endif
