#ifndef TATOOINE_FLOWEXPLORER_UI_PIN_H
#define TATOOINE_FLOWEXPLORER_UI_PIN_H
//==============================================================================
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/flowexplorer/ui/pinkind.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
struct base_node;
//==============================================================================
struct pin {
 private:
  ax::NodeEditor::PinId m_id;
  std::string           m_name;
  base_node&        m_node;
  pinkind               m_kind;
  std::type_info const& m_type;

 public:
  pin(base_node& n, std::type_info const& type, pinkind kind,
      std::string const& name)
      : m_id{boost::hash<boost::uuids::uuid>{}(
            boost::uuids::random_generator()())},
        m_name{name},
        m_node{n},
        m_kind{kind},
        m_type{type} {}

  auto id() const {
    return m_id;
  }
  auto node() const -> auto const& {
    return m_node;
  }
  auto node() -> auto& {
    return m_node;
  }
  auto name() -> auto& {
    return m_name;
  }
  auto name() const -> auto const& {
    return m_name;
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
auto make_input_pin(base_node& n, std::string const& name) {
  return pin{n, typeid(std::decay_t<T>), pinkind::input, name};
}
//------------------------------------------------------------------------------
template <typename T>
auto make_output_pin(base_node& n, std::string const& name) {
  return pin{n, typeid(std::decay_t<T>), pinkind::output, name};
}
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#include "node.h"
#endif
