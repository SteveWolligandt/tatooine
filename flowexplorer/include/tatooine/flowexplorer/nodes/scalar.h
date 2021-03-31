#ifndef TATOOINE_FLOWEXPLORER_NODES_SCALAR_H
#define TATOOINE_FLOWEXPLORER_NODES_SCALAR_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct scalar : ui::node<scalar> {
  real_t m_value          = 0;
  real_t m_internal_value = 0;
  real_t m_speed          = 0.01;
  bool   m_vary           = false;
  int    m_variation_type = 0;

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
    TATOOINE_REFLECTION_INSERT_METHOD(internal_value, m_internal_value),
    TATOOINE_REFLECTION_INSERT_METHOD(speed, m_speed),
    TATOOINE_REFLECTION_INSERT_METHOD(vary, m_vary),
    TATOOINE_REFLECTION_INSERT_METHOD(variation_type, m_variation_type));
#endif
