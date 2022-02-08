#ifndef TATOOINE_FLOWEXPLORER_NODES_SCALAR_H
#define TATOOINE_FLOWEXPLORER_NODES_SCALAR_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct scalar : ui::node<scalar> {
  real_type m_value          = 0;
  real_type m_speed          = 0.01;
  bool   m_vary           = false;

  scalar(flowexplorer::scene& s);
  virtual ~scalar() = default;
  auto draw_properties() -> bool override;
  auto update(std::chrono::duration<double> const&) -> void override;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::scalar,
    TATOOINE_REFLECTION_INSERT_METHOD(value, m_value),
    TATOOINE_REFLECTION_INSERT_METHOD(speed, m_speed),
    TATOOINE_REFLECTION_INSERT_METHOD(vary, m_vary));
#endif
