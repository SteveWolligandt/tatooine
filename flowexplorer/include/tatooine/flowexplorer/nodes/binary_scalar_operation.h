#ifndef TATOOINE_FLOWEXPLORER_NODES_BINARY_SCALAR_OPERATION_H
#define TATOOINE_FLOWEXPLORER_NODES_BINARY_SCALAR_OPERATION_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct binary_scalar_operation : ui::node<binary_scalar_operation> {
  enum class op : int { addition, subtraction, multiplication, division };
  real_t         m_value = 0;
  int            m_op    = 0;
  ui::input_pin& m_s0;
  ui::input_pin& m_s1;

  binary_scalar_operation(flowexplorer::scene& s);
  virtual ~binary_scalar_operation() = default;
  auto draw_properties() -> bool override;
  auto on_property_changed() -> void override;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::binary_scalar_operation,
    TATOOINE_REFLECTION_INSERT_METHOD(operation, m_op));
#endif
