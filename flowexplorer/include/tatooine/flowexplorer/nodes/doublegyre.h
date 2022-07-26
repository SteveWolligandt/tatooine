#ifndef TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE_H
#define TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct doublegyre : tatooine::analytical::numerical::doublegyre<real_type>,
                    ui::node<doublegyre> {
  doublegyre(flowexplorer::scene& s)
      : ui::node<doublegyre>{
            "Double Gyre", s,
            *dynamic_cast<polymorphic::vectorfield<real_type, 2>*>(this)} {
    this->set_infinite_domain(true);
  }
  virtual ~doublegyre() = default;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(tatooine::flowexplorer::nodes::doublegyre);
#endif
