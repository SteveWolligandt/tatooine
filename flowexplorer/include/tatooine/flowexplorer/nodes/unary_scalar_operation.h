#ifndef TATOOINE_FLOWEXPLORER_NODES_UNARY_SCALAR_OPERATION_H
#define TATOOINE_FLOWEXPLORER_NODES_UNARY_SCALAR_OPERATION_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct unary_scalar_operation : ui::node<unary_scalar_operation> {
  enum class op : int { sin, cos };
  real_t         m_value = 0;
  int            m_op    = 0;
  ui::input_pin& m_input;

  unary_scalar_operation(flowexplorer::scene& s);
  virtual ~unary_scalar_operation() = default;
  auto draw_properties() -> bool override;
  auto on_property_changed() -> void override;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::unary_scalar_operation,
    TATOOINE_REFLECTION_INSERT_METHOD(operation, m_op));
#endif
