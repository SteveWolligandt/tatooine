#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/ui/link.h>
#include <tatooine/flowexplorer/ui/node.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
pin::pin(base::node& n, std::type_info const& type, pinkind kind,
         std::string const& title)
    : m_title{title}, m_node{n}, m_kind{kind}, m_type{type} {}
//------------------------------------------------------------------------------
auto pin::add_link(struct link& l) -> void {
  m_links.push_back(&l);
  m_node.on_pin_connected(*this, this == &l.input() ? l.output() : l.input());
}
//------------------------------------------------------------------------------
auto pin::remove_link(struct link& l) -> void {
  auto it = std::find(begin(m_links), end(m_links), &l);
  m_links.erase(it);
  m_node.on_pin_disconnected(*this);
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
