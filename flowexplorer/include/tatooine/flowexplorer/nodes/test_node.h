#ifndef TATOOINE_FLOWEXPLORER_NODES_TEST_NODE_H
#define TATOOINE_FLOWEXPLORER_NODES_TEST_NODE_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/reflection.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct test_node : ui::node<test_node> {
 private:
  //============================================================================
  float m_test_var;

 public:
  //============================================================================
  // CONSTRUCTORS
  //============================================================================
  test_node(flowexplorer::scene& s)
      : node<test_node>{"Test", s}, m_test_var{1.0f} {}

 public:
  //============================================================================
  // SETTER / GETTER
  //============================================================================
  auto test_var() -> auto& { return m_test_var; } auto test_var() const {
  return m_test_var;
}
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
REGISTER_NODE(tatooine::flowexplorer::nodes::test_node,
              TATOOINE_REFLECTION_INSERT_GETTER(test_var));
#endif
