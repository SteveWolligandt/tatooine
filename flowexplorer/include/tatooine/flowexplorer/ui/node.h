#ifndef TATOOINE_FLOWEXPLORER_UI_NODE_H
#define TATOOINE_FLOWEXPLORER_UI_NODE_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/uuid_holder.h>
#include <tatooine/flowexplorer/serializable.h>
namespace tatooine::flowexplorer {
struct scene;
}
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
struct node : uuid_holder<ax::NodeEditor::NodeId>, serializable {
 private:
  std::string            m_title;
  scene const&           m_scene;
  std::vector<pin>       m_input_pins;
  std::vector<pin>       m_output_pins;

 public:
  node(scene const& s);
  node(std::string const& title, scene const& s);
  virtual ~node() = default;
  //============================================================================
  template <typename T>
  auto insert_input_pin(std::string const& title) {
    m_input_pins.push_back(make_input_pin<T>(*this, title));
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto insert_output_pin(std::string const& title) {
    m_output_pins.push_back(make_output_pin<T>(*this, title));
  }
  //----------------------------------------------------------------------------
  auto title() const -> auto const& {
    return m_title;
  }
  //----------------------------------------------------------------------------
  auto title() -> auto& {
    return m_title;
  }
  //----------------------------------------------------------------------------
  auto set_title(std::string const& title) {
    m_title = title;
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
    ed::BeginNode(get_id());
    ImGui::TextUnformatted(title().c_str());
    f();
    for (auto& input_pin : m_input_pins) {
      ed::BeginPin(input_pin.get_id(), ed::PinKind::Input);
      std::string in = "-> " + input_pin.title();
      ImGui::TextUnformatted(in.c_str());
      ed::EndPin();
    }
    for (auto& output_pin : m_output_pins) {
      ed::BeginPin(output_pin.get_id(), ed::PinKind::Output);
      std::string out = output_pin.title() + " ->";
      ImGui::TextUnformatted(out.c_str());
      ed::EndPin();
    }
    ed::EndNode();
  }
  //----------------------------------------------------------------------------
  auto                   node_position() const -> ImVec2;
  virtual void           on_pin_connected(pin& this_pin, pin& other_pin) {}
  virtual void on_pin_disconnected(pin& this_pin) {}
  constexpr virtual auto node_type_name() const -> std::string_view = 0;
};
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
#endif
