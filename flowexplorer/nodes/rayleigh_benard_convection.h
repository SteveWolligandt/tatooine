#ifndef TATOOINE_FLOWEXPLORER_NODES_RAYLEIGH_BENARD_CONVECTION_H
#define TATOOINE_FLOWEXPLORER_NODES_RAYLEIGH_BENARD_CONVECTION_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/rayleigh_benard_convection.h>
#include <yavin>
#include "../renderable.h"
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct rayleigh_benard_convection
    : tatooine::analytical::fields::numerical::rayleigh_benard_convection<Real>,
      renderable {
  rayleigh_benard_convection(struct window& w)
      : renderable{w, "Rayleigh Benard Convection"} {
    this->template insert_output_pin<parent::field<Real, 3, 3>>("Field Out");
  }
  void render(const yavin::mat4&, const yavin::mat4&) override {}
  void draw_ui() override {
    ui::node::draw_ui([this] {
      ImGui::SliderDouble("A", &this->A());
      ImGui::SliderDouble("k", &this->k());
    });
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
