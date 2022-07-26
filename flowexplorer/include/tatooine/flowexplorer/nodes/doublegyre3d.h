#ifndef TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE3D_H
#define TATOOINE_FLOWEXPLORER_NODES_DOUBLEGYRE3D_H
//==============================================================================
#include <tatooine/analytical/numerical/doublegyre3d.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct doublegyre3d
    : tatooine::analytical::numerical::doublegyre3d<real_type>,
      ui::node<doublegyre3d> {
  doublegyre3d(flowexplorer::scene& s)
      : ui::node<doublegyre3d>{
            "Double Gyre 3D", s,
            *dynamic_cast<polymorphic::vectorfield<real_type, 3>*>(this)} {}
  virtual ~doublegyre3d() = default;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(tatooine::flowexplorer::nodes::doublegyre3d);
#endif
