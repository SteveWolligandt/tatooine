#ifndef TATOOINE_FLOWEXPLORER_NODES_DUFFING_OSCILLATOR_H
#define TATOOINE_FLOWEXPLORER_NODES_DUFFING_OSCILLATOR_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/duffing_oscillator.h>
#include "../renderable.h"
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct duffing_oscillator
    : tatooine::analytical::fields::numerical::duffing_oscillator<Real>,
      renderable {
  duffing_oscillator(struct window& w)
      : renderable{w, "Duffing Oscillator"},
        tatooine::analytical::fields::numerical::duffing_oscillator<Real>{
            Real(1), Real(1), Real(1)} {
    this->template insert_output_pin<parent::vectorfield<Real, 2>>("Field Out");
  }
  void render(mat<float, 4, 4> const&, mat<float, 4, 4> const&) override {}
  void draw_ui() override {
    ui::node::draw_ui([this] {
      ImGui::DragDouble("alpha", &this->m_alpha);
      ImGui::DragDouble("beta", &this->m_beta);
      ImGui::DragDouble("delt", &this->m_delta);
    });
  }
  auto is_transparent() const -> bool override {
    return false;
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
