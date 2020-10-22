#ifndef TATOOINE_FLOWEXPLORER_UI_NODE_H
#define TATOOINE_FLOWEXPLORER_UI_NODE_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/demangling.h>
#include <tatooine/flowexplorer/uuid_holder.h>
#include <tatooine/flowexplorer/serializable.h>
#include <tatooine/reflection.h>
#include <tatooine/vec.h>
#include <yavin/imgui.h>
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
template <typename T>
struct node_serializer {
  //----------------------------------------------------------------------------
  auto serialize(T const& t) const -> toml::table {
    toml::table serialized_node;
    reflection::for_each(t, [&serialized_node](auto const& name,
                                               auto const& var) {
      if constexpr (std::is_arithmetic_v<std::decay_t<decltype(var)>>) {
        serialized_node.insert(name, var);
      } else if constexpr (
          std::is_same_v<std::array<int, 2>, std::decay_t<decltype(var)>> ||
          std::is_same_v<std::array<float, 2>, std::decay_t<decltype(var)>> ||
          std::is_same_v<std::array<double, 2>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<int, 2>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<float, 2>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<double, 2>, std::decay_t<decltype(var)>>) {
        serialized_node.insert(name, toml::array{var.at(0), var.at(1)});
      } else if constexpr (
          std::is_same_v<std::array<int, 3>, std::decay_t<decltype(var)>> ||
          std::is_same_v<std::array<float, 3>, std::decay_t<decltype(var)>> ||
          std::is_same_v<std::array<double, 3>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<int, 3>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<float, 3>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<double, 3>, std::decay_t<decltype(var)>>) {
        serialized_node.insert(name,
                               toml::array{var.at(0), var.at(1), var.at(2)});
      } else if constexpr (
          std::is_same_v<std::array<int, 4>, std::decay_t<decltype(var)>> ||
          std::is_same_v<std::array<float, 4>, std::decay_t<decltype(var)>> ||
          std::is_same_v<std::array<double, 4>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<int, 4>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<float, 4>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<double, 4>, std::decay_t<decltype(var)>>) {
        serialized_node.insert(
            name, toml::array{var.at(0), var.at(1), var.at(2), var.at(3)});
      }
    });
    return serialized_node;
  }
  //----------------------------------------------------------------------------
  auto deserialize(T& t, toml::table const& serialized_node) -> void {
    reflection::for_each(t, [&serialized_node](auto const& name, auto& var) {
      if constexpr (std::is_integral_v<std::decay_t<decltype(var)>>) {
        var = serialized_node[name].as_integer()->get();
      } else if constexpr (std::is_floating_point_v<
                               std::decay_t<decltype(var)>>) {
        var = serialized_node[name].as_floating_point()->get();
      } else if constexpr (std::is_same_v<std::array<int, 2>,
                                          std::decay_t<decltype(var)>> ||
                           std::is_same_v<vec<int, 2>,
                                          std::decay_t<decltype(var)>>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_integer()->get();
        var.at(1) = arr[1].as_integer()->get();
      } else if constexpr (std::is_same_v<std::array<int, 3>,
                                          std::decay_t<decltype(var)>> ||
                           std::is_same_v<vec<int, 3>,
                                          std::decay_t<decltype(var)>>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_integer()->get();
        var.at(1) = arr[1].as_integer()->get();
        var.at(2) = arr[2].as_integer()->get();
      } else if constexpr (std::is_same_v<std::array<int, 4>,
                                          std::decay_t<decltype(var)>> ||
                           std::is_same_v<vec<int, 4>,
                                          std::decay_t<decltype(var)>>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_integer()->get();
        var.at(1) = arr[1].as_integer()->get();
        var.at(2) = arr[2].as_integer()->get();
        var.at(3) = arr[3].as_integer()->get();

      } else if constexpr (
          std::is_same_v<std::array<float, 2>, std::decay_t<decltype(var)>> ||
          std::is_same_v<std::array<double, 2>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<float, 2>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<double, 2>, std::decay_t<decltype(var)>>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_floating_point()->get();
        var.at(1) = arr[1].as_floating_point()->get();
      } else if constexpr (
          std::is_same_v<std::array<float, 3>, std::decay_t<decltype(var)>> ||
          std::is_same_v<std::array<double, 3>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<float, 3>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<double, 3>, std::decay_t<decltype(var)>>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_floating_point()->get();
        var.at(1) = arr[1].as_floating_point()->get();
        var.at(2) = arr[2].as_floating_point()->get();
      } else if constexpr (
          std::is_same_v<std::array<float, 4>, std::decay_t<decltype(var)>> ||
          std::is_same_v<std::array<double, 4>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<float, 4>, std::decay_t<decltype(var)>> ||
          std::is_same_v<vec<double, 4>, std::decay_t<decltype(var)>>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_floating_point()->get();
        var.at(1) = arr[1].as_floating_point()->get();
        var.at(2) = arr[2].as_floating_point()->get();
        var.at(3) = arr[3].as_floating_point()->get();
      }
    });
  }
  //----------------------------------------------------------------------------
  auto draw_ui(T& t) -> void {
    reflection::for_each(t, [](auto const& name, auto& var) {
      // float
      if constexpr (std::is_same_v<float, std::decay_t<decltype(var)>>) {
        ImGui::DragFloat(name, &var, 0.1f);
      } else if constexpr (std::is_same_v<std::array<float, 2>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragFloat2(name, var.data(), 0.1f);
      } else if constexpr (std::is_same_v<std::array<float, 3>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragFloat3(name, var.data(), 0.1f);
      } else if constexpr (std::is_same_v<std::array<float, 4>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragFloat4(name, var.data(), 0.1f);
      } else if constexpr (std::is_same_v<vec<float, 2>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragFloat2(name, var.data_ptr(), 0.1f);
      } else if constexpr (std::is_same_v<vec<float, 3>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragFloat3(name, var.data_ptr(), 0.1f);
      } else if constexpr (std::is_same_v<vec<float, 4>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragFloat4(name, var.data_ptr(), 0.1f);

        // double
      } else if constexpr (std::is_same_v<double,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragDouble(name, &var, 0.1);
      } else if constexpr (std::is_same_v<std::array<double, 2>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragDouble2(name, var.data(), 0.1);
      } else if constexpr (std::is_same_v<std::array<double, 3>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragDouble3(name, var.data(), 0.1);
      } else if constexpr (std::is_same_v<std::array<double, 4>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragDouble4(name, var.data(), 0.1);
      } else if constexpr (std::is_same_v<vec<double, 2>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragDouble2(name, var.data_ptr(), 0.1);
      } else if constexpr (std::is_same_v<vec<double, 3>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragDouble3(name, var.data_ptr(), 0.1);
      } else if constexpr (std::is_same_v<vec<double, 4>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragDouble4(name, var.data_ptr(), 0.1);

        // int
      } else if constexpr (std::is_same_v<int, std::decay_t<decltype(var)>>) {
        ImGui::DragInt(name, &var, 1);
      } else if constexpr (std::is_same_v<std::array<int, 2>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragInt2(name, var.data(), 1);
      } else if constexpr (std::is_same_v<std::array<int, 3>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragInt3(name, var.data(), 1);
      } else if constexpr (std::is_same_v<std::array<int, 4>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragInt4(name, var.data(), 1);
      } else if constexpr (std::is_same_v<vec<int, 2>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragInt2(name, var.data_ptr(), 1);
      } else if constexpr (std::is_same_v<vec<int, 3>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragInt3(name, var.data_ptr(), 1);
      } else if constexpr (std::is_same_v<vec<int, 4>,
                                          std::decay_t<decltype(var)>>) {
        ImGui::DragInt4(name, var.data_ptr(), 1);
      }
    });
  }
  //----------------------------------------------------------------------------
  constexpr auto type_name() const -> std::string_view {
    return reflection::name<T>();
  }
};
template <typename Child>
struct node : base::node, node_serializer<Child> {
  using base::node::node;
  using serializer_t = node_serializer<Child>;
  //============================================================================
  auto serialize() const -> toml::table override final {
    return serializer_t::serialize(*dynamic_cast<Child const*>(this));
  }
  //----------------------------------------------------------------------------
  auto deserialize(toml::table const& serialized_node) -> void override final {
    return node_serializer<Child>::deserialize(*dynamic_cast<Child*>(this),
                                               serialized_node);
  }
  //----------------------------------------------------------------------------
  auto draw_ui() -> void override final {
    return node_serializer<Child>::draw_ui(*dynamic_cast<Child*>(this));
  }
  //----------------------------------------------------------------------------
  auto type_name() const -> std::string_view override final {
    return node_serializer<Child>::type_name();
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

#define REGISTER_NODE_FACTORY(registered_function_, sec)           \
    static constexpr registered_function_t ptr_##registered_function_          \
        __attribute((used, section(#sec))) = {                                 \
          .registered_function = registered_function_,             \
        }                                                                      \
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
  REGISTER_NODE_FACTORY(register_node, registration);                          \
  }                                                                            \
  TATOOINE_MAKE_ADT_REFLECTABLE(type, ##__VA_ARGS__)                           
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
