#ifndef TATOOINE_FLOWEXPLORER_UI_NODE_H
#define TATOOINE_FLOWEXPLORER_UI_NODE_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <boost/hana.hpp>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/uuid_holder.h>
#include <tatooine/flowexplorer/serializable.h>
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
  flowexplorer::scene*                 m_scene;
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
};
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
struct registered_function_t {
  using registered_function_ptr_t =
      tatooine::flowexplorer::ui::base::node* (*)(tatooine::flowexplorer::scene&,
                                            std::string const&);
  registered_function_ptr_t registered_function;
};

#define REGISTER_NODE_FACTORY(registered_function_, sec)                       \
  static constexpr registered_function_t ptr_##registered_function_            \
      __attribute((used, section(#sec))) = {                                   \
          .registered_function = registered_function_,                         \
  }
//------------------------------------------------------------------------------
#define REGISTER_NODE(type)                                                    \
  static auto register_##type(tatooine::flowexplorer::scene& s,                \
                              std::string const&             node_type_name)   \
      ->tatooine::flowexplorer::ui::base::node* {                              \
    if (node_type_name == #type) {                                             \
      return new type{s};                                                      \
    }                                                                          \
    return nullptr;                                                            \
  }                                                                            \
  REGISTER_NODE_FACTORY(register_##type, registration)
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
//==============================================================================
#define BEGIN_META_NODE(type)                                                  \
  namespace boost::hana {                                                      \
  template <>                                                                  \
  struct accessors_impl<type> {                                                \
    static BOOST_HANA_CONSTEXPR_LAMBDA auto apply() {                          \
    return make_tuple(

#define END_META_NODE() );}};}

#define META_NODE_ACCESSOR(name, accessor)                                     \
  make_pair(BOOST_HANA_STRING(#name), [](auto&& p) -> decltype(auto) {         \
    return p.accessor;                                                         \
  })

#endif

