#include <tatooine/flowexplorer/ui/link.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/scene.h>
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
auto input_pin::link(struct link& l) -> void {
  if (m_link != nullptr) {
    node().on_pin_disconnected(*this);
  }
  m_link = &l;
  node().on_pin_connected(*this, l.output());
}
//------------------------------------------------------------------------------
auto input_pin::unlink() -> void {
  if (is_linked()) {
    m_link->output().unlink_from_input(*m_link);
    node().on_pin_disconnected(*this);
    node().scene().remove_link(*m_link);
    m_link = nullptr;
  }
}
//------------------------------------------------------------------------------
auto input_pin::unlink_from_output() -> void {
  if (is_linked()) {
    node().on_pin_disconnected(*this);
  }
  m_link = nullptr;
}
//------------------------------------------------------------------------------
auto input_pin::linked_type() const -> std::type_info const& {
  return link().output().type();
}
//------------------------------------------------------------------------------
auto output_pin::link(struct link& l) -> void {
  m_links.push_back(&l);
  node().on_pin_connected(*this, l.input());
}
//------------------------------------------------------------------------------
auto output_pin::unlink(struct link& l) -> void {
  auto it = std::find(begin(m_links), end(m_links), &l);
  if (it != end(m_links)) {
    (*it)->input().unlink_from_output();
    node().scene().remove_link(**it);
    node().on_pin_disconnected(*this);
    m_links.erase(it);
  }
}
//------------------------------------------------------------------------------
auto output_pin::unlink_all() -> void {
  for (auto& l : m_links) {
    l->input().unlink_from_output();
    node().on_pin_disconnected(*this);
  }
  for (auto& l : m_links) {
    node().scene().remove_link(*l);
  }
  m_links.clear();
}
//------------------------------------------------------------------------------
auto output_pin::unlink_from_input(struct link& l) -> void {
  auto it = std::find(begin(m_links), end(m_links), &l);
  if (it != end(m_links)) {
    node().on_pin_disconnected(*this);
    m_links.erase(it);
  }
}
//------------------------------------------------------------------------------
auto output_pin::set_active(bool active) -> void {
  toggleable::set_active(active);
  if (!active) {
    unlink_all();
  }
}
//------------------------------------------------------------------------------
auto input_pin::set_active(bool active) -> void {
  toggleable::set_active(active);
  if (!active) {
    unlink();
  }
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
