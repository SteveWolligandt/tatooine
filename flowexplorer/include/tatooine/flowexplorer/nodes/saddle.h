#ifndef TATOOINE_FLOWEXPLORER_NODES_SADDLE_H
#define TATOOINE_FLOWEXPLORER_NODES_SADDLE_H
//==============================================================================
#include <tatooine/analytical/numerical/saddle.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct saddle : tatooine::analytical::numerical::saddle<double>,
                ui::node<saddle> {
  saddle(flowexplorer::scene& s)
      : ui::node<saddle>{"Saddle Field", s,
                         typeid(polymorphic::vectorfield<double, 2>)} {}
  virtual ~saddle() = default;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(tatooine::flowexplorer::nodes::saddle)
#endif
