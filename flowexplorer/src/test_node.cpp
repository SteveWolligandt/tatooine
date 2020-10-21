#include <tatooine/flowexplorer/nodes/test_node.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
test_node::test_node(flowexplorer::scene& s)
    : node<test_node>{"Test", s},
      m_test_var{1.0f} {
}
//==============================================================================
auto test_node::draw_ui() -> void {
   ImGui::DragFloat("var", &m_test_var, 0.1, 0.0f, 1.0f);
}
//----------------------------------------------------------------------------
auto test_node::on_pin_connected(ui::pin& this_pin, ui::pin& other_pin)
    -> void {}
//----------------------------------------------------------------------------
auto test_node::on_pin_disconnected(ui::pin& this_pin) -> void {}
//----------------------------------------------------------------------------
auto test_node::serialize() const -> toml::table {
  toml::table serialized_node;
  serialized_node.insert("test_var", m_test_var);


  return serialized_node;
}
//----------------------------------------------------------------------------
void test_node::deserialize(toml::table const& serialization) {
  m_test_var = serialization["test_var"].as_floating_point()->get();
}
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================