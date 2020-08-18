#ifndef TATOOINE_FLOWEXPLORER_UI_NODE_H
#define TATOOINE_FLOWEXPLORER_UI_NODE_H
//==============================================================================
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "../imgui-node-editor/imgui_node_editor.h"
#include "pin.h"
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
struct node {
 private:
  ax::NodeEditor::NodeId m_id;
  std::vector<pin>       m_input_pins;
  std::vector<pin>       m_output_pins;

 public:
  node()
      : m_id{boost::hash<boost::uuids::uuid>{}(
            boost::uuids::random_generator()())} {}
  //============================================================================
  template <typename T>
  auto insert_input_pin() {
    m_input_pins.push_back(make_input_pin<T>(*this));
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto insert_output_pin() {
    m_output_pins.push_back(make_output_pin<T>(*this));
  }
  //----------------------------------------------------------------------------
  auto id() {
    return m_id;
  }
  //----------------------------------------------------------------------------
  auto input_pins() const -> auto const& {
    return m_input_pins;
  }
  //----------------------------------------------------------------------------
  auto input_pins() -> auto& {
    return m_input_pins;
  }
  //----------------------------------------------------------------------------
  auto output_pins() const -> auto const& {
    return m_output_pins;
  }
  //----------------------------------------------------------------------------
  auto output_pins() -> auto& {
    return m_output_pins;
  }
  //----------------------------------------------------------------------------
  template <typename F>
  void draw_ui(F&& f) {
    namespace ed = ax::NodeEditor;
    ed::BeginNode(m_id);
    ImGui::Text("Node");
    f();
    for (auto& input_pin : m_input_pins) {
      ed::BeginPin(input_pin.id(), ed::PinKind::Input);
      ImGui::Text("-> In");
      ed::EndPin();
    }
    for (auto& output_pin : m_output_pins) {
      ed::BeginPin(output_pin.id(), ed::PinKind::Output);
      ImGui::Text("Out ->");
      ed::EndPin();
    }
    ed::EndNode();
  }
  virtual void on_pin_connected(pin& this_pin, pin& other_pin) {}
  virtual void on_pin_disconnected(pin& this_pin) {}
};
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#endif
