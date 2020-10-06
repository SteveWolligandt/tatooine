#ifndef TATOOINE_FLOWEXPLORER_NODES_SPACETIME_VECTORFIELD_H
#define TATOOINE_FLOWEXPLORER_NODES_SPACETIME_VECTORFIELD_H
//==============================================================================
#include <tatooine/spacetime_vectorfield.h>
#include <tatooine/flowexplorer/renderable.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct spacetime_vectorfield
    : tatooine::spacetime_vectorfield<parent::vectorfield<Real, 2> const*, Real, 3>,
      ui::node {
  spacetime_vectorfield()
      : tatooine::spacetime_vectorfield<parent::vectorfield<Real, 2> const*,
                                        Real, 3>{nullptr},
        ui::node{"Space-Time Vector Field"} {
    this->template insert_input_pin<parent::vectorfield<Real, 2>>(
        "2D Vector Field");
    this->template insert_output_pin<parent::vectorfield<Real, 3>>("3D Vector Field");
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
