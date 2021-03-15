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
  distance_measure(flowexplorer::scene& s)
      : ui::node<distance_measure>{"Distance Measure", s},
        m_x0_pin{insert_input_pin<vec2>("x0")},
        m_x1_pin{insert_input_pin<vec2>("x1")} {}
  //----------------------------------------------------------------------------
  virtual ~distance_measure() = default;
  //============================================================================
  auto draw_properties() -> bool override {
    if (m_x0_pin.is_connected() && m_x1_pin.is_connected()) {
      ImGui::Text("distance = %f", distance(m_x0_pin.linked_object_as<vec2>(),
                                            m_x1_pin.linked_object_as<vec2>()));
    }
    return false;
  }
};
//==============================================================================
};  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::distance_measure)
#endif
