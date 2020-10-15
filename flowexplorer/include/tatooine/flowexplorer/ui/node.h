#ifndef TATOOINE_FLOWEXPLORER_UI_NODE_H
#define TATOOINE_FLOWEXPLORER_UI_NODE_H
//==============================================================================
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/serializable.h>
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
struct base_node : serializable {
 private:
  ax::NodeEditor::NodeId m_id;
  std::string            m_name;
  std::vector<pin>       m_input_pins;
  std::vector<pin>       m_output_pins;

 public:
  base_node(std::string const& name)
      : m_name{name},
        m_id{boost::hash<boost::uuids::uuid>{}(
            boost::uuids::random_generator()())} {}
  virtual ~base_node() = default;
  //============================================================================
  template <typename T>
  auto insert_input_pin(std::string const& name) {
    m_input_pins.push_back(make_input_pin<T>(*this, name));
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto insert_output_pin(std::string const& name) {
    m_output_pins.push_back(make_output_pin<T>(*this, name));
  }
  //----------------------------------------------------------------------------
  auto id() {
    return m_id;
  }
  //----------------------------------------------------------------------------
  auto name() const -> auto const& {
    return m_name;
  }
  //----------------------------------------------------------------------------
  auto name() -> auto& {
    return m_name;
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
  virtual void draw_ui() {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename F>
  void draw_ui(F&& f) {
    namespace ed = ax::NodeEditor;
    ed::BeginNode(m_id);
    ImGui::TextUnformatted(name().c_str());
    f();
    for (auto& input_pin : m_input_pins) {
      ed::BeginPin(input_pin.id(), ed::PinKind::Input);
      std::string in = "-> " + input_pin.name();
      ImGui::TextUnformatted(in.c_str());
      ed::EndPin();
    }
    for (auto& output_pin : m_output_pins) {
      ed::BeginPin(output_pin.id(), ed::PinKind::Output);
      std::string out = output_pin.name() + " ->";
      ImGui::TextUnformatted(out.c_str());
      ed::EndPin();
    }
    ed::EndNode();
  }
  auto node_position() const -> ImVec2 {
    return ax::NodeEditor::GetNodePosition(m_id);
  }
  virtual void on_pin_connected(pin& this_pin, pin& other_pin) {}
  virtual void on_pin_disconnected(pin& this_pin) {}
  constexpr virtual auto node_type_name() const -> std::string_view = 0;
};
template <typename Derived>
struct node : base_node {
  using base_node::base_node;
  virtual ~node() = default;
  //virtual constexpr auto node_type_name() -> std::string_view {
  //  return Derived::node_type_name();
  //}
  //constexpr auto node_type_name_() const -> std::string_view override {
  //  return node_type_name();
  //}
};
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#endif
