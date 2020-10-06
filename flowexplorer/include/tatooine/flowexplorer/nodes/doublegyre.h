#ifndef TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE_H
#define TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct doublegyre : tatooine::analytical::fields::numerical::doublegyre<Real>,
                    ui::node {
  doublegyre() : ui::node{"Double Gyre"} {
    this->set_infinite_domain(true);
    this->template insert_output_pin<parent::vectorfield<Real, 2>>("Field Out");
  }
  virtual ~doublegyre() = default;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
