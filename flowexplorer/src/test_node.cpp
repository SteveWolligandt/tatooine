#include <tatooine/flowexplorer/nodes/test_node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
test_node::test_node(flowexplorer::scene& s)
    : node<test_node>{"Test", s},
      m_test_var{1.0f} {
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
