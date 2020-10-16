#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/flowexplorer/scene.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
node::node(std::string const& title, scene const& s)
    : m_id{boost::hash<boost::uuids::uuid>{}(
          boost::uuids::random_generator()())},
      m_title{title},
      m_scene{s} {}
//------------------------------------------------------------------------------
node::node(scene const& s)
    : m_id{boost::hash<boost::uuids::uuid>{}(
          boost::uuids::random_generator()())},
      m_title{""},
      m_scene{s} {}
//------------------------------------------------------------------------------
auto node::node_position() const -> ImVec2 {
  ImVec2 pos;
  m_scene.do_in_context([&] {
    pos = ax::NodeEditor::GetNodePosition(m_id);
  });
  return pos;
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
