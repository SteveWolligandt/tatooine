#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/nodes/distance_measure.h>
//============================================================================
namespace tatooine::flowexplorer::nodes {
//============================================================================
distance_measure::distance_measure(flowexplorer::scene& s)
    : ui::node<distance_measure>{"Distance Measure", s},
      m_x0_pin{insert_input_pin<vec2>("x0")},
      m_x1_pin{insert_input_pin<vec2>("x1")} {}
//============================================================================
auto distance_measure::draw_properties() -> bool {
  if (m_x0_pin.is_linked() && m_x1_pin.is_linked()) {
    ImGui::Text("distance = %f", distance(m_x0_pin.get_linked_as<vec2>(),
                                          m_x1_pin.get_linked_as<vec2>()));
  }
  return false;
}
//============================================================================
}  // namespace tatooine::flowexplorer::nodes
//============================================================================
