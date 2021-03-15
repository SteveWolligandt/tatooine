#ifndef TATOOINE_FLOWEXPLORER_NODES_RAYLEIGH_BENARD_CONVECTION_H
#define TATOOINE_FLOWEXPLORER_NODES_RAYLEIGH_BENARD_CONVECTION_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/rayleigh_benard_convection.h>
#include <tatooine/flowexplorer/renderable.h>

#include <yavin>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct rayleigh_benard_convection
    : tatooine::analytical::fields::numerical::rayleigh_benard_convection<double>,
      ui::node<rayleigh_benard_convection> {
  rayleigh_benard_convection(flowexplorer::scene& s)
      : ui::node<rayleigh_benard_convection>{"Rayleigh Benard Convection", s} {
    this->template insert_output_pin<parent::field<double, 3, 3>>("Field Out",
                                                                  *this);
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::rayleigh_benard_convection,
    TATOOINE_REFLECTION_INSERT_GETTER(A), TATOOINE_REFLECTION_INSERT_GETTER(k))
#endif
