#ifndef TATOOINE_FLOWEXPLORER_NODES_SPACETIME_SPLITTED_VECTORFIELD_H
#define TATOOINE_FLOWEXPLORER_NODES_SPACETIME_SPLITTED_VECTORFIELD_H
//==============================================================================
#include <tatooine/spacetime_splitted_vectorfield.h>
#include <tatooine/flowexplorer/renderable.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct spacetime_splitted_vectorfield
    : tatooine::spacetime_splitted_vectorfield<
          parent::vectorfield<double, 3> const*>,
      ui::node<spacetime_splitted_vectorfield> {
  spacetime_splitted_vectorfield(flowexplorer::scene& s)
      : tatooine::spacetime_splitted_vectorfield<
            parent::vectorfield<double, 3> const*>{nullptr},
        ui::node<spacetime_splitted_vectorfield>{
            "Space-Time Splitted Vector Field", s} {
    this->template insert_input_pin<parent::vectorfield<double, 3>>(
        "3D Vector Field");
    this->template insert_output_pin<parent::vectorfield<double, 2>>(
        "2D Vector Field", *this);
  }
  //----------------------------------------------------------------------------
  auto on_pin_connected(ui::input_pin& /*this_pin*/, ui::output_pin& other_pin)
      -> void override {
    this->set_field(
        dynamic_cast<parent::vectorfield<double, 3> const*>(&other_pin.node()));
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::spacetime_splitted_vectorfield);
#endif
