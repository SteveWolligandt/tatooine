#ifndef TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE_H
#define TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include "../renderable.h"
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct doublegyre : tatooine::analytical::fields::numerical::doublegyre<Real>, renderable {
  doublegyre(struct window& w) : renderable{w, "Double Gyre"} {
    this->template insert_output_pin<parent::vectorfield<Real, 2>>("Field Out");
  }
  void render(mat<float, 4, 4> const&, mat<float, 4, 4> const&) override {}
  void draw_ui() override {
    ui::node::draw_ui([this] {});
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
