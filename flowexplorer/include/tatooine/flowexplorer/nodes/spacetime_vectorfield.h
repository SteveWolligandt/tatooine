#ifndef TATOOINE_FLOWEXPLORER_NODES_SPACETIME_VECTORFIELD_H
#define TATOOINE_FLOWEXPLORER_NODES_SPACETIME_VECTORFIELD_H
//==============================================================================
#include <tatooine/spacetime_vectorfield.h>
#include <tatooine/flowexplorer/renderable.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct spacetime_vectorfield
    : tatooine::spacetime_vectorfield<parent::vectorfield<double, 2> const*, double, 3>,
      ui::node<spacetime_vectorfield> {
  spacetime_vectorfield(flowexplorer::scene& s)
      : tatooine::spacetime_vectorfield<parent::vectorfield<double, 2> const*,
                                        double, 3>{nullptr},
        ui::node<spacetime_vectorfield>{"Space-Time Vector Field", s} {
    this->template insert_input_pin<parent::vectorfield<double, 2>>(
        "2D Vector Field");
    this->template insert_output_pin<parent::vectorfield<double, 3>>(
        "3D Vector Field");
  }
  //----------------------------------------------------------------------------
  void on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) override {
    if (this_pin.kind() == ui::pinkind::input) {
      this->set_field(dynamic_cast<parent::vectorfield<double, 2> const*>(
          &other_pin.node()));
    }
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(tatooine::flowexplorer::nodes::spacetime_vectorfield);
#endif
