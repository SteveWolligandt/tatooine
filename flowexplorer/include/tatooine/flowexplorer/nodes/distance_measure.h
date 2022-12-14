#ifndef TATOOINE_FLOWEXPLORER_NODES_DISTANCE_MEASURE_H
#define TATOOINE_FLOWEXPLORER_NODES_DISTANCE_MEASURE_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct distance_measure : ui::node<distance_measure> {
  ui::input_pin& m_x0_pin;
  ui::input_pin& m_x1_pin;
  distance_measure(flowexplorer::scene& s);
  virtual ~distance_measure() = default;
  //============================================================================
  auto draw_properties() -> bool override;
};
//==============================================================================
};  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::distance_measure)
#endif
