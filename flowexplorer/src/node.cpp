#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::ui::base {
//==============================================================================
node::node(std::string const& title, flowexplorer::scene& s)
    : m_title{title}, m_scene{&s} {}
//------------------------------------------------------------------------------
node::node(flowexplorer::scene& s) : m_title{""}, m_scene{&s} {}
//------------------------------------------------------------------------------
auto node::node_position() const -> ImVec2 {
  ImVec2 pos;
  m_scene->do_in_context(
      [&] { pos = ax::NodeEditor::GetNodePosition(get_id()); });
  return pos;
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui::base
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
auto insert_registered_element(scene& s, std::string_view const& name) {
  iterate_registered_factories(factory) {
    if (auto ptr = factory->f(s, node_type_name); ptr) {
      return ptr;
    }
  }
  return nullptr;
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
