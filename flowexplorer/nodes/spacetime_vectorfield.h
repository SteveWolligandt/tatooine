#ifndef TATOOINE_FLOWEXPLORER_NODES_SPACETIME_VECTORFIELD_H
#define TATOOINE_FLOWEXPLORER_NODES_SPACETIME_VECTORFIELD_H
//==============================================================================
#include <tatooine/spacetime_vectorfield.h>
#include "../renderable.h"
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct spacetime_vectorfield
    : tatooine::spacetime_vectorfield<parent::vectorfield<Real, 2> const*, Real, 3>,
      renderable {
  spacetime_vectorfield(struct window& w)
      : tatooine::spacetime_vectorfield<parent::vectorfield<Real, 2> const*,
                                        Real, 3>{nullptr},
        renderable{w, "Space-Time Vector Field"} {
    this->template insert_input_pin<parent::vectorfield<Real, 2>>("2D Vector Field");
    this->template insert_output_pin<parent::vectorfield<Real, 3>>("3D Vector Field");
  }
  void render(const yavin::mat4&, const yavin::mat4&) override {}
  void draw_ui() override {
    ui::node::draw_ui([this] {});
  }
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) override {
    if (this_pin.kind() == ui::pinkind::input) {
      this->set_field(
          dynamic_cast<parent::vectorfield<Real, 2> const*>(&other_pin.node()));
    }
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
