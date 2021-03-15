#ifndef TATOOINE_FLOWEXPLORER_NODES_DISTANCE_MEASURE_H
#define TATOOINE_FLOWEXPLORER_NODES_DISTANCE_MEASURE_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct distance_measure : ui::node<distance_measure> {
  distance_measure(flowexplorer::scene& s)
      : ui::node<distance_measure>{"Distance Measure", s} {
    insert_input_pin<vec2>("x0");
    insert_input_pin<vec2>("x1");
  }
  virtual ~distance_measure() = default;
  //============================================================================
  auto draw_properties() -> bool override {
    if (input_pins()[0].is_connected() && input_pins()[1].is_connected()) {
      ImGui::Text("distance = %f",
                  distance(input_pins()[0].link().output().get<vec2>(),
                           input_pins()[1].link().output().get<vec2>()));
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
