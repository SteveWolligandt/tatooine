#ifndef TATOOINE_FLOWEXPLORER_NODES_RAYLEIGH_BENARD_CONVECTION_H
#define TATOOINE_FLOWEXPLORER_NODES_RAYLEIGH_BENARD_CONVECTION_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/rayleigh_benard_convection.h>
#include <tatooine/flowexplorer/renderable.h>

#include <yavin>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct rayleigh_benard_convection
    : tatooine::analytical::fields::numerical::rayleigh_benard_convection<Real>,
      ui::node {
  rayleigh_benard_convection() : ui::node{"Rayleigh Benard Convection"} {
    this->template insert_output_pin<parent::field<Real, 3, 3>>("Field Out");
  }
  void draw_ui() override {
    ui::node::draw_ui([this] {
      ImGui::DragDouble("A", &this->A());
      ImGui::DragDouble("k", &this->k());
    });
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
