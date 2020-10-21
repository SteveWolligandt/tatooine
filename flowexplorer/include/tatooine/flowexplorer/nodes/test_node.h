#ifndef TATOOINE_FLOWEXPLORER_NODES_TEST_NODE_H
#define TATOOINE_FLOWEXPLORER_NODES_TEST_NODE_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct test_node : ui::node<test_node> {
 private:
  //============================================================================
  float       m_test_var;

 public:
  //============================================================================
  // CONSTRUCTORS
  //============================================================================
  test_node(flowexplorer::scene& s);

 public:
  //============================================================================
  // NODE OVERRIDES
  //============================================================================
  auto draw_ui() -> void override;
  //----------------------------------------------------------------------------
  auto on_pin_connected(ui::pin& this_pin, ui::pin& other_pin) -> void override;
  //----------------------------------------------------------------------------
  auto on_pin_disconnected(ui::pin& this_pin) -> void override;
  //----------------------------------------------------------------------------
  auto serialize() const -> toml::table override;
  //----------------------------------------------------------------------------
  auto deserialize(toml::table const& serialization) -> void override;
  //----------------------------------------------------------------------------
  constexpr auto node_type_name() const -> std::string_view override {
    return "test_node";
  }

 public:
  //============================================================================
  // SETTER / GETTER
  //============================================================================
  auto test_var() -> auto& { return m_test_var; }
  auto test_var() const { return m_test_var; }
};
REGISTER_NODE(test_node);
static constexpr non_const_method_ptr<tatooine::flowexplorer::nodes::test_node,
                                      float&>
    test_node_test_var_method =
        &tatooine::flowexplorer::nodes::test_node::test_var;
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
#endif
