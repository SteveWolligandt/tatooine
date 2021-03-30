#ifndef TATOOINE_FLOWEXPLORER_UI_LINK_H
#define TATOOINE_FLOWEXPLORER_UI_LINK_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/flowexplorer/uuid_holder.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
struct output_pin;
struct input_pin;
//==============================================================================
using link_uuid = uuid_holder<ax::NodeEditor::LinkId>;
struct link : link_uuid {
 private:
  input_pin*  m_input_pin;
  output_pin* m_output_pin;

 public:
  //============================================================================
  link(size_t const id, input_pin& in, output_pin& out)
      : uuid_holder<ax::NodeEditor::LinkId>{id},
        m_input_pin{&in},
        m_output_pin{&out} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  link(input_pin& in, output_pin& out)
      : m_input_pin{&in}, m_output_pin{&out} {}
  //============================================================================
  auto input() const -> auto const& {
    return *m_input_pin;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto input() -> auto& {
    return *m_input_pin;
  }
  //----------------------------------------------------------------------------
  auto output() const -> auto const& {
    return *m_output_pin;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto output() -> auto& {
    return *m_output_pin;
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#endif
