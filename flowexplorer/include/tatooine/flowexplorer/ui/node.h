#ifndef TATOOINE_FLOWEXPLORER_UI_NODE_H
#define TATOOINE_FLOWEXPLORER_UI_NODE_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/uuid_holder.h>
#include <tatooine/flowexplorer/serializable.h>
#include <tatooine/reflection.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
struct scene;
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
namespace tatooine::flowexplorer::ui {
//==============================================================================
namespace base {
struct node : uuid_holder<ax::NodeEditor::NodeId>, serializable {
 private:
  std::string            m_title;
  flowexplorer::scene*   m_scene;
  std::vector<pin>       m_input_pins;
  std::vector<pin>       m_output_pins;

 public:
  node(flowexplorer::scene & s);
  node(std::string const& title, flowexplorer::scene & s);
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
  virtual void draw_ui() = 0;
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
  auto         node_position() const -> ImVec2;
  virtual auto on_pin_connected(pin& this_pin, pin& other_pin) -> void {}
  virtual auto on_pin_disconnected(pin& this_pin) -> void {}
  virtual auto type_name() const -> std::string_view = 0;

  auto scene() const -> auto const& {
    return *m_scene;
  }
  auto scene() -> auto & {
    return *m_scene;
  }
};
}  // namespace base
template <typename Child>
struct node : base::node {
  using base::node::node;
  //----------------------------------------------------------------------------
  auto serialize() const -> toml::table override {
    toml::table serialized_node;
    reflection::for_each(
        *dynamic_cast<Child const*>(this),
        [&serialized_node](auto const& name, auto const& var) {
          if constexpr (std::is_same_v<float, std::decay_t<decltype(var)>>) {
            serialized_node.insert(name, var);
          }
        });
    return serialized_node;
  }
  //----------------------------------------------------------------------------
  auto deserialize(toml::table const& serialization) -> void override {
    reflection::for_each(
        *dynamic_cast<Child*>(this),
        [&serialization](auto const& name, auto& var) {
          if constexpr (std::is_same_v<float, std::decay_t<decltype(var)>>) {
            var = serialization[name].as_floating_point()->get();
          }
        });
  }
  //----------------------------------------------------------------------------
  auto draw_ui() -> void override {
    reflection::for_each(
        *dynamic_cast<Child*>(this), [](auto const& name, auto& var) {
          if constexpr (std::is_same_v<float, std::decay_t<decltype(var)>>) {
            ImGui::DragFloat(name, &var, 0.1f);
          }
        });
  }
  //----------------------------------------------------------------------------
  auto type_name() const -> std::string_view override {
    return reflection::name<Child>();
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
struct registered_function_t {
  using registered_function_ptr_t =
      tatooine::flowexplorer::ui::base::node* (*)(tatooine::flowexplorer::scene&,
                                            std::string_view const&);
  registered_function_ptr_t registered_function;
};

#define REGISTER_NODE_FACTORY(namespace_, registered_function_, sec)           \
  static constexpr registered_function_t ptr_##registered_function_            \
      __attribute((used, section(#sec))) = {                                   \
          .registered_function = namespace_::registered_function_,             \
  }
//------------------------------------------------------------------------------
#define REGISTER_NODE(type, ...)                                               \
  namespace tatooine::flowexplorer::registered_funcs::type {                   \
  auto register_node(::tatooine::flowexplorer::scene& s,                       \
                     std::string_view const&          node_type_name)          \
      -> ::tatooine::flowexplorer::ui::base::node* {                           \
    if (node_type_name == #type) {                                             \
      return new ::type{s};                                                    \
    }                                                                          \
    return nullptr;                                                            \
  }                                                                            \
  }                                                                            \
  REGISTER_NODE_FACTORY(tatooine::flowexplorer::registered_funcs::type,        \
                        register_node, registration);                          \
  TATOOINE_MAKE_ADT_REFLECTABLE(type, __VA_ARGS__)
//------------------------------------------------------------------------------
extern registered_function_t __start_registration;
extern registered_function_t __stop_registration;
#define iterate_registered_functions(elem, section)                            \
  for (registered_function_t* elem = &__start_##section;                       \
       elem != &__stop_##section; ++elem)
//------------------------------------------------------------------------------
#define call_registered_functions(section, string)                             \
  iterate_registered_functions(entry, section) {                               \
    entry->registered_function();                                              \
  }
#endif
