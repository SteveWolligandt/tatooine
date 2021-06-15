#include <tatooine/flowexplorer/scene.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/ui/link.h>
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/field.h>
#include <tatooine/flowexplorer/nodes/axis_aligned_bounding_box.h>
#include <tatooine/flowexplorer/nodes/position.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
auto icon_color(std::type_info const& type, float const alpha) -> ImVec4 {
  if (type == typeid(polymorphic::vectorfield<double, 2>) ||
      type == typeid(polymorphic::vectorfield<double, 3>)) {
    return {1, 0.5, 0.5, alpha};
  } else if (type == typeid(nodes::axis_aligned_bounding_box<2>) ||
             type == typeid(nodes::axis_aligned_bounding_box<3>)) {
    return {0.5, 1, 0.5, alpha};
  } else if (type == typeid(nodes::position<2>) ||
             type == typeid(nodes::position<3>)) {
    return {0.5, 0.5, 1, alpha};
  }
  return {1, 1, 1, alpha};
}
//------------------------------------------------------------------------------
auto icon_color(ui::output_pin const& pin, float const alpha) -> ImVec4 {
  if (pin.type() == typeid(polymorphic::vectorfield<double, 2>) ||
      pin.type() == typeid(polymorphic::vectorfield<double, 3>)) {
    return icon_color(typeid(polymorphic::vectorfield<double, 2>), alpha);
  } else if (pin.type() == typeid(nodes::axis_aligned_bounding_box<2>) ||
             pin.type() == typeid(nodes::axis_aligned_bounding_box<3>)) {
    return icon_color(typeid(nodes::axis_aligned_bounding_box<2>), alpha);
  } else if (pin.type() == typeid(nodes::position<2>) ||
             pin.type() == typeid(nodes::position<3>)) {
    return icon_color(typeid(nodes::position<2>), alpha);
  }
  return {1, 1, 1, alpha};
}
//------------------------------------------------------------------------------
auto icon_color(ui::input_pin const& pin, float const alpha) -> ImVec4 {
  if (std::any_of(begin(pin.types()), end(pin.types()), [](auto t) {
        return *t == typeid(polymorphic::vectorfield<double, 2>) ||
               *t == typeid(polymorphic::vectorfield<double, 3>);
      })) {
    return icon_color(typeid(polymorphic::vectorfield<double, 2>), alpha);
  } else if (std::any_of(begin(pin.types()), end(pin.types()), [](auto t) {
               return *t == typeid(nodes::axis_aligned_bounding_box<2>) ||
                      *t == typeid(nodes::axis_aligned_bounding_box<3>);
             })) {
    return icon_color(typeid(nodes::axis_aligned_bounding_box<2>), alpha);
  } else if (std::any_of(begin(pin.types()), end(pin.types()), [](auto t) {
               return *t == typeid(nodes::position<2>) ||
                      *t == typeid(nodes::position<3>);
             })) {
    return icon_color(typeid(nodes::position<2>), alpha);
  }
  return {1, 1, 1, alpha};
}
//------------------------------------------------------------------------------
base_pin::base_pin(base::node& n, pinkind kind, std::string const& title,
                   icon_type t)
    : m_title{title}, m_node{n}, m_kind{kind}, m_icon_type{t} {}
//------------------------------------------------------------------------------
auto base_pin::draw(size_t const icon_size, float const alpha) const -> void {
  icon(ImVec2(icon_size, icon_size), m_icon_type, is_linked(),
       ImVec4(1, 1, 1, alpha));
}
//------------------------------------------------------------------------------
input_pin::input_pin(base::node& n, std::vector<std::type_info const*> types,
                     std::string const& title, icon_type const t)
    : base_pin{n, pinkind::input, title, t}, m_types{types} {}
//------------------------------------------------------------------------------
output_pin::output_pin(base::node& n, std::string const& title,
                       icon_type const t)
    : base_pin{n, pinkind::output, title, t} {}
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
