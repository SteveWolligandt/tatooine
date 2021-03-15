#include <tatooine/flowexplorer/ui/link.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/flowexplorer/ui/pin.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
base_pin::base_pin(base::node& n, pinkind kind, std::string const& title)
    : m_title{title}, m_node{n}, m_kind{kind} {}
//------------------------------------------------------------------------------
input_pin::input_pin(base::node& n, std::vector<std::type_info const*> types,
                     std::string const& title)
    : base_pin{n, pinkind::input, title}, m_types{types} {}
//------------------------------------------------------------------------------
output_pin::output_pin(base::node& n, std::string const& title)
    : base_pin{n, pinkind::output, title} {}
//------------------------------------------------------------------------------
auto input_pin::set_link(struct link& l) -> void {
  if (m_link != nullptr) {
    node().on_pin_disconnected(*this);
  }
  m_link = &l;
  node().on_pin_connected(*this, l.output());
}
//------------------------------------------------------------------------------
auto input_pin::unset_link() -> void {
  m_link = nullptr;
  node().on_pin_disconnected(*this);
}
//------------------------------------------------------------------------------
auto output_pin::insert_link(struct link& l) -> void {
  m_links.push_back(&l);
  node().on_pin_connected(*this, l.input());
}
//------------------------------------------------------------------------------
auto output_pin::remove_link(struct link& l) -> void {
  auto it = std::find(begin(m_links), end(m_links), &l);
  m_links.erase(it);
  node().on_pin_disconnected(*this);
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
