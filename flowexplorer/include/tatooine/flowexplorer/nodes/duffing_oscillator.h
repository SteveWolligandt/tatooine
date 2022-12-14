#ifndef TATOOINE_FLOWEXPLORER_NODES_DUFFING_OSCILLATOR_H
#define TATOOINE_FLOWEXPLORER_NODES_DUFFING_OSCILLATOR_H
//==============================================================================
#include <tatooine/analytical/numerical/duffing_oscillator.h>
#include <tatooine/flowexplorer/renderable.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct duffing_oscillator
    : tatooine::analytical::numerical::duffing_oscillator<double>,
      ui::node<duffing_oscillator> {
  duffing_oscillator(flowexplorer::scene& s)
      : tatooine::analytical::numerical::duffing_oscillator<
            double>{1.0, 1.0, 1.0},
        ui::node<duffing_oscillator>{"Duffing Oscillator", s} {
    this->template insert_output_pin<polymorphic::vectorfield<double, 2>>(
        "Field Out", *this);
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(tatooine::flowexplorer::nodes::duffing_oscillator,
              TATOOINE_REFLECTION_INSERT_GETTER(alpha),
              TATOOINE_REFLECTION_INSERT_GETTER(beta),
              TATOOINE_REFLECTION_INSERT_GETTER(delta))
#endif
