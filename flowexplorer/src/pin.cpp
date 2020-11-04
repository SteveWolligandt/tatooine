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
auto pin::set_link(struct link& l) -> void {
  m_link = &l;
  m_node.on_pin_connected(*this, this == &l.input() ? l.output() : l.input());
}
//------------------------------------------------------------------------------
auto pin::unset_link() -> void {
  m_link = nullptr;
  m_node.on_pin_disconnected(*this);
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
