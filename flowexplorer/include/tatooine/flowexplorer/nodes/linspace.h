#ifndef TATOOINE_FLOWEXPLORER_NODES_LINSPACE_H
#define TATOOINE_FLOWEXPLORER_NODES_LINSPACE_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/linspace.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct linspace : ui::node<linspace>, tatooine::linspace<real_t> {
  linspace(flowexplorer::scene& s);
  virtual ~linspace() = default;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
TATOOINE_FLOWEXPLORER_REGISTER_NODE(
    tatooine::flowexplorer::nodes::linspace
    , TATOOINE_REFLECTION_INSERT_METHOD(front, front())
    , TATOOINE_REFLECTION_INSERT_METHOD(back, back())
    , TATOOINE_REFLECTION_INSERT_METHOD(size, size())
    );
#endif
