#ifndef TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
#define TATOOINE_FLOWEXPLORER_NODES_ABCFLOW_H
//==============================================================================
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct abcflow : tatooine::analytical::fields::numerical::abcflow<double>,
                 ui::node<abcflow> {
  abcflow(flowexplorer::scene& s)
      : ui::node<abcflow>{"ABC Flow", s,
                          typeid(parent::vectorfield<double, 3>)} {}
  virtual ~abcflow() = default;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(tatooine::flowexplorer::nodes::abcflow);
#endif
