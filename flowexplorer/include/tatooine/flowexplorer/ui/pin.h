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
struct pin : uuid_holder<ax::NodeEditor::PinId>{
 private:
  std::string           m_title;
  base::node&           m_node;
  pinkind               m_kind;
  std::type_info const& m_type;
  struct link*          m_link = nullptr;

 public:
  pin(base::node& n, std::type_info const& type, pinkind kind,
      std::string const& title);

  auto node() const -> auto const& {
    return m_node;
  }
  auto node() -> auto& {
    return m_node;
  }
  auto title() -> auto& {
    return m_title;
  }
  auto title() const -> auto const& {
    return m_title;
  }
  auto kind() const {
    return m_kind;
  }
  auto type() const -> auto const& {
    return m_type;
  }
  auto is_connected() const { return m_link != nullptr; }
  auto set_link(struct link& l) -> void;
  auto unset_link() ->void;
  auto link() const -> auto const& { return *m_link; }
  auto link() -> auto& { return *m_link; }
};
//==============================================================================
template <typename T>
auto make_input_pin(base::node& n, std::string const& title) {
  return pin{n, typeid(std::decay_t<T>), pinkind::input, title};
}
//------------------------------------------------------------------------------
template <typename T>
auto make_output_pin(base::node& n, std::string const& title) {
  return pin{n, typeid(std::decay_t<T>), pinkind::output, title};
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#include "node.h"
#endif
