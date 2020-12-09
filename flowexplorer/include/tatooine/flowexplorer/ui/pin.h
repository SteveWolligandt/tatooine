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
  struct link*                 m_link;

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
  std::type_info const&     m_type;
  std::vector<struct link*> m_links;

 public:
  output_pin(base::node& n, std::type_info const& type,
             std::string const& title);

  auto insert_link(struct link & l) -> void;
  auto remove_link(struct link & l) -> void;
  auto type() const -> auto const& { return m_type; }
  auto is_connected() const { return !m_links.empty(); }
  auto links() const -> auto const& { return m_links; }
  auto links() -> auto& { return m_links; }
};
//==============================================================================
template <typename... Ts>
auto make_input_pin(base::node& n, std::string const& title) {
  return input_pin{n, std::vector{&typeid(std::decay_t<Ts>)...}, title};
}
//------------------------------------------------------------------------------
template <typename T>
auto make_output_pin(base::node& n, std::string const& title) {
  return output_pin{n, typeid(std::decay_t<T>), title};
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#include "node.h"
#endif
