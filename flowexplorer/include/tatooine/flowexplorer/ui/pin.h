#ifndef TATOOINE_FLOWEXPLORER_UI_PIN_H
#define TATOOINE_FLOWEXPLORER_UI_PIN_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
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
struct base_pin : uuid_holder<ax::NodeEditor::PinId> {
 private:
  std::string m_title;
  base::node& m_node;
  pinkind     m_kind;

 public:
  base_pin(base::node& n, pinkind kind, std::string const& title);

  auto node() const -> auto const& { return m_node; }
  auto node() -> auto& { return m_node; }
  auto title() -> auto& { return m_title; }
  auto title() const -> auto const& { return m_title; }
  auto kind() const { return m_kind; }
};
//==============================================================================
struct input_pin : base_pin {
 private:
  std::vector<std::type_info const*> m_types;
  struct link*                 m_link = nullptr;

 public:
  input_pin(base::node& n, std::vector<std::type_info const*> types,
            std::string const& title);

  auto types() const -> auto const& { return m_types; }
  auto is_connected() const { return m_link != nullptr; }
  auto link() const -> auto const& { return *m_link; }
  auto link() -> auto& { return *m_link; }
  auto set_link(struct link & l) -> void;
  auto unset_link() -> void;
};
//==============================================================================
struct output_pin : base_pin {
 private:
  std::vector<struct link*> m_links;

 public:
  output_pin(base::node& n, std::string const& title);
  virtual ~output_pin() = default;

  auto insert_link(struct link & l) -> void;
  auto remove_link(struct link & l) -> void;
  auto is_connected() const { return !m_links.empty(); }
  auto links() const -> auto const& { return m_links; }
  auto links() -> auto& { return m_links; }
  virtual auto type() const -> std::type_info const& = 0;

  template <typename T>
  auto get() -> T&;
};
template <typename T>
struct output_pin_impl : output_pin {
 private:
  T& m_ref;

 public:
  output_pin_impl(base::node& n, std::string const& title, T& ref)
      : output_pin{n, title}, m_ref{ref} {}
  virtual ~output_pin_impl() = default;
  auto type() const -> std::type_info const& override {
    return typeid(T);
  }

  auto get() -> T& { return m_ref; }
  auto get() const -> T const& { return m_ref; }
};
//==============================================================================
template <typename T>
auto output_pin::get() -> T& {
  if (typeid(T) != this->type()) {
    throw std::runtime_error{"Types do not match."};
  }
  return dynamic_cast<output_pin_impl<T>*>(this)->get();
}
//==============================================================================
template <typename... Ts>
auto make_input_pin(base::node& n, std::string const& title) {
  return input_pin{n, std::vector{&typeid(std::decay_t<Ts>)...}, title};
}
//------------------------------------------------------------------------------
template <typename T>
auto make_output_pin(base::node& n, std::string const& title, T& t)
    -> std::unique_ptr<output_pin> {
  return std::make_unique<output_pin_impl<T>>(n, title, t);
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#include "node.h"
#endif
