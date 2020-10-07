#ifndef TATOOINE_FLOWEXPLORER_NODES_SADDLE_H
#define TATOOINE_FLOWEXPLORER_NODES_SADDLE_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
template <typename Real>
struct saddle : tatooine::analytical::fields::numerical::saddle<Real>,
                    ui::node {
  saddle() : ui::node{"Saddle Field"} {
    this->template insert_output_pin<parent::vectorfield<Real, 2>>("Field Out");
  }
  virtual ~saddle() = default;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
