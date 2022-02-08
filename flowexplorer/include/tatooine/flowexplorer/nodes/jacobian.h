#ifndef TATOOINE_FLOWEXPLORER_NODES_JACOBIAN_H
#define TATOOINE_FLOWEXPLORER_NODES_JACOBIAN_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/differentiated_field.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes{
//==============================================================================
template <size_t N>
struct jacobian : ui::node<jacobian<N>>,
                  differentiated_field<polymorphic::vectorfield<real_type, N>*> {
  jacobian(flowexplorer::scene& s)
      : ui::node<jacobian<N>>{
            "Jacobian", s,
            *dynamic_cast<polymorphic::matrixfield<real_type, N>*>(this)} {
    this->template insert_input_pin<polymorphic::vectorfield<real_type, N>>("V");
  }
  auto on_pin_connected(ui::input_pin& p, ui::output_pin&) -> void override {
    this->set_internal_field(&p.get_linked_as<polymorphic::vectorfield<real_type, N>>());
  }
};
//==============================================================================
using jacobian2 = jacobian<2>;
using jacobian3 = jacobian<3>;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(tatooine::flowexplorer::nodes::jacobian2);
TATOOINE_FLOWEXPLORER_REGISTER_NODE(tatooine::flowexplorer::nodes::jacobian3);
#endif
