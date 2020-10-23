#ifndef TATOOINE_FLOWEXPLORER_UI_LINK_H
#define TATOOINE_FLOWEXPLORER_UI_LINK_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/flowexplorer/uuid_holder.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
struct link : uuid_holder<ax::NodeEditor::LinkId> {
 private:
  pin* m_input_pin;
  pin* m_output_pin;

 public:
  //============================================================================
  link(size_t const id, pin& input_pin, pin& output_pin)
      : uuid_holder<ax::NodeEditor::LinkId>{id},
        m_input_pin{&input_pin},
        m_output_pin{&output_pin} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  link(pin& input_pin, pin& output_pin)
      : m_input_pin{&input_pin}, m_output_pin{&output_pin} {}
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
