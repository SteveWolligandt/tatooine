#ifndef TATOOINE_FLOWEXPLORER_UI_PIN_H
#define TATOOINE_FLOWEXPLORER_UI_PIN_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/flowexplorer/toggleable.h>
#include <tatooine/flowexplorer/ui/draw_icon.h>
#include <tatooine/flowexplorer/ui/link.h>
#include <tatooine/flowexplorer/ui/pinkind.h>
#include <tatooine/flowexplorer/uuid_holder.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
namespace base {
struct node;
}
struct link;
//==============================================================================
struct pin : uuid_holder<ax::NodeEditor::PinId>, toggleable {
 private:
  std::string m_title;
  base::node& m_node;
  pinkind     m_kind;
  icon_type   m_icon_type;

 public:
  pin(base::node& n, pinkind kind, std::string const& title,
           icon_type const t = icon_type::flow);

  auto node() const -> auto const& { return m_node; }
  auto node() -> auto& { return m_node; }
  auto title() -> auto& { return m_title; }
  auto title() const -> auto const& { return m_title; }
  auto kind() const { return m_kind; }
  auto draw(size_t const icon_size, float const alpha) const -> void;
  auto set_icon_type(icon_type const t) { m_icon_type = t; }

  virtual auto is_linked() const -> bool = 0;
};
//==============================================================================
struct output_pin;
struct input_pin : pin {
  friend struct output_pin;
 private:
  std::vector<std::type_info const*> m_types;
  struct link*                       m_link = nullptr;

 public:
  input_pin(base::node& n, std::vector<std::type_info const*> types,
            std::string const& title, icon_type const t);
  virtual ~input_pin() = default;

  auto types() const -> auto const& { return m_types; }
  auto is_linked() const -> bool override { return m_link != nullptr; }
  auto link() const -> auto const& { return *m_link; }
  auto link() -> auto& { return *m_link; }
  auto link(struct link & l) -> void;
  auto unlink() -> void;
 private:
  auto unlink_from_output() -> void;
 public:
  template <typename T>
  auto get_linked_as() -> T&;
  template <typename T>
  auto get_linked_as() const -> T const&;
  auto linked_type() const -> std::type_info const&;
  auto set_active(bool active = true) -> void override;
};
//==============================================================================
struct input_pin_property_link {
  input_pin & m_pin;
  input_pin_property_link(input_pin& pin) : m_pin{pin} {}
  virtual ~input_pin_property_link() = default;
  virtual auto update() -> bool      = 0;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Prop>
struct input_pin_property_link_impl : input_pin_property_link {
  Prop& m_property;
  input_pin_property_link_impl(input_pin& pin, Prop& prop)
      : input_pin_property_link{pin}, m_property{prop} {}
  virtual ~input_pin_property_link_impl() = default;
  virtual auto update() -> bool {
    bool changed = false;
    if (m_pin.is_linked()) {
      auto const& linked = m_pin.get_linked_as<Prop>();

      if (linked != m_property) {
        m_property = linked;
        changed    = true;
      }
    }
    return changed;
  }
};
//==============================================================================
struct output_pin : pin {
  friend struct input_pin;
 private:
  std::vector<struct link*> m_links;

 public:
  output_pin(base::node& n, std::string const& title, icon_type const t);
  virtual ~output_pin() = default;

  auto link(struct link & l) -> void;
  auto unlink(struct link& l) -> void;
  auto unlink_all() -> void;

 private:
  auto unlink_from_input(struct link & l) -> void;
 public:
  auto is_linked() const -> bool override { return !m_links.empty(); }
  auto links() const -> auto const& { return m_links; }
  auto links() -> auto& { return m_links; }
  virtual auto type() const -> std::type_info const& = 0;

  template <typename T>
  auto get_as() -> T&;

  auto set_active(bool active = true) -> void override;
};
//==============================================================================
template <typename T>
struct output_pin_impl : output_pin {
 private:
  T& m_ref;

 public:
  output_pin_impl(base::node& n, std::string const& title, T& ref,
                  icon_type const t)
      : output_pin{n, title, t}, m_ref{ref} {}
  virtual ~output_pin_impl() = default;
  auto type() const -> std::type_info const& override {
    return typeid(T);
  }

  auto get_as() -> T& { return m_ref; }
  auto get_as() const -> T const& { return m_ref; }
};
//==============================================================================
template <typename T>
auto output_pin::get_as() -> T& {
  if (typeid(T) != this->type()) {
    throw std::runtime_error{"Types do not match."};
  }
  return dynamic_cast<output_pin_impl<T>*>(this)->get_as();
}
//==============================================================================
template <typename T>
auto input_pin::get_linked_as() -> T& {
  return link().output().get_as<T>();
}
//------------------------------------------------------------------------------
template <typename T>
auto input_pin::get_linked_as() const -> T const& {
  return link().output().get_as<T>();
}
//==============================================================================
template <typename... Ts>
auto make_input_pin(base::node& n, std::string const& title,
                    icon_type const t = icon_type::flow)
    -> std::unique_ptr<input_pin> {
  return std::make_unique<input_pin>(
      n, std::vector{&typeid(std::decay_t<Ts>)...}, title, t);
}
//------------------------------------------------------------------------------
template <typename Prop>
auto make_input_pin_property_link(input_pin& pin, Prop& prop)
    -> std::unique_ptr<input_pin_property_link> {
  return std::make_unique<input_pin_property_link_impl<Prop>>(pin, prop);
}
//------------------------------------------------------------------------------
template <typename T>
auto make_output_pin(base::node& n, std::string const& title, T& t,
                    icon_type const it = icon_type::flow)
    -> std::unique_ptr<output_pin> {
  return std::make_unique<output_pin_impl<T>>(n, title, t, it);
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#include "node.h"
#endif
