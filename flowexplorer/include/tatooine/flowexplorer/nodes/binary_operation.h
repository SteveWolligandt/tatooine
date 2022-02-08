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
  template <typename Lhs, typename Rhs, typename Op>
  using op_field = binary_operation_field<Lhs, Rhs, Op>;

  template <size_t LhsN, size_t RhsN, typename Op>
  using op_field_vec_vec =
      binary_operation_field<polymorphic::vectorfield<real_type, LhsN>*,
                             polymorphic::vectorfield<real_type, RhsN>*, Op>;
  template <size_t LhsN, size_t RhsN, typename Op>
  using op_field_mat_vec =
      binary_operation_field<polymorphic::matrixfield<real_type, LhsN>*,
                             polymorphic::vectorfield<real_type, RhsN>*, Op>;

  enum class operation_t : int {
    addition,
    subtraction,
    multiplication,
    division,
    dot
  };
  static constexpr auto mult = [](auto const& lhs, auto const& rhs) {
    return lhs * rhs;
  };
  static constexpr auto dot = [](auto const& lhs, auto const& rhs) {
    return tatooine::dot(lhs, rhs);
  };

  template <size_t N>
  using mat_vec_mult_field_t  = op_field_mat_vec<N, N, decltype(mult)>;
  using mat_vec_mult_field2_t = mat_vec_mult_field_t<2>;
  using mat_vec_mult_field3_t = mat_vec_mult_field_t<3>;
  template <size_t N>
  using dot_field_t  = op_field_vec_vec<N, N, decltype(dot)>;
  using dot_field2_t = dot_field_t<2>;
  using dot_field3_t = dot_field_t<3>;

  // output data
  std::variant<std::monostate,
               real_type,
               dot_field2_t,
               dot_field3_t,
               mat_vec_mult_field2_t,
               mat_vec_mult_field3_t>
      m_output_data;

  int             m_operation = 0;
  ui::input_pin&  m_input0;
  ui::input_pin&  m_input1;
  ui::output_pin& m_scalar_pin_out;
  ui::output_pin& m_dot_field2_pin_out;
  ui::output_pin& m_dot_field3_pin_out;
  ui::output_pin& m_mat_vec_mult_field2_pin_out;
  ui::output_pin& m_mat_vec_mult_field3_pin_out;

  binary_operation(flowexplorer::scene& s);
  virtual ~binary_operation() = default;
  auto draw_properties() -> bool override;
  auto on_property_changed() -> void override;
  auto on_pin_connected(ui::input_pin&, ui::output_pin&) -> void override;
  auto on_pin_disconnected(ui::input_pin&) -> void override;
  auto deactivate_output_pins() -> void;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::binary_operation,
    TATOOINE_REFLECTION_INSERT_METHOD(scalar_operation, m_operation));
#endif
