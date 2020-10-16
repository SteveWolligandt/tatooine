#ifndef TATOOINE_FLOWEXPLORER_UI_PIN_H
#define TATOOINE_FLOWEXPLORER_UI_PIN_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/flowexplorer/ui/pinkind.h>
#include <tatooine/flowexplorer/uuid_holder.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
struct node;
//==============================================================================
struct pin : uuid_holder<ax::NodeEditor::PinId>{
 private:
  std::string           m_title;
  node&                 m_node;
  pinkind               m_kind;
  std::type_info const& m_type;

 public:
  pin(node& n, std::type_info const& type, pinkind kind,
      std::string const& title)
      : m_title{title},
        m_node{n},
        m_kind{kind},
        m_type{type} {}

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
};
//==============================================================================
template <typename T>
auto make_input_pin(node& n, std::string const& title) {
  return pin{n, typeid(std::decay_t<T>), pinkind::input, title};
}
//------------------------------------------------------------------------------
template <typename T>
auto make_output_pin(node& n, std::string const& title) {
  return pin{n, typeid(std::decay_t<T>), pinkind::output, title};
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#include "node.h"
#endif
