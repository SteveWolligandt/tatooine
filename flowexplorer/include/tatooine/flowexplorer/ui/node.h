#ifndef TATOOINE_FLOWEXPLORER_UI_NODE_H
#define TATOOINE_FLOWEXPLORER_UI_NODE_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/demangling.h>
#include <tatooine/flowexplorer/serializable.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/uuid_holder.h>
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
struct node : uuid_holder<ax::NodeEditor::NodeId>, serializable  {
 private:
  std::string          m_title;
  flowexplorer::scene* m_scene;
  std::vector<input_pin>  m_input_pins;
  std::vector<output_pin> m_output_pins;
  bool                 m_enabled = true;

 public:
  node(flowexplorer::scene& s);
  node(std::string const& title, flowexplorer::scene& s);
  virtual ~node() = default;
  //============================================================================
  template <typename... Ts>
  auto insert_input_pin(std::string const& title) {
    m_input_pins.push_back(make_input_pin<Ts...>(*this, title));
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto insert_output_pin(std::string const& title) {
    m_output_pins.push_back(make_output_pin<T>(*this, title));
  }
  //----------------------------------------------------------------------------
  auto title() const -> auto const& { return m_title; }
  auto title() -> auto& { return m_title; }
  //----------------------------------------------------------------------------
  auto scene() const -> auto const& { return *m_scene; }
  auto scene() -> auto& { return *m_scene; }
  //----------------------------------------------------------------------------
  auto is_enabled() const { return m_enabled; }
  //----------------------------------------------------------------------------
  auto enable(bool en = true) -> void { m_enabled = en; }
  auto disable() -> void { m_enabled = false; }
  auto toggle() -> void { m_enabled = !m_enabled; }
  //----------------------------------------------------------------------------
  auto set_title(std::string const& title) { m_title = title; }
  //----------------------------------------------------------------------------
  auto input_pins() const -> auto const& { return m_input_pins; }
  auto input_pins() -> auto& { return m_input_pins; }
  //----------------------------------------------------------------------------
  auto output_pins() const -> auto const& { return m_output_pins; }
  auto output_pins() -> auto& { return m_output_pins; }
  //----------------------------------------------------------------------------
  virtual auto draw_properties() -> bool = 0;
  virtual auto on_property_changed() -> void {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto draw_node() -> void;
  //----------------------------------------------------------------------------
  auto         node_position() const -> ImVec2;
  virtual auto on_pin_connected(input_pin& /*this_pin*/,
                                output_pin& /*other_pin*/) -> void {}
  virtual auto on_pin_connected(output_pin& /*this_pin*/,
                                input_pin& /*other_pin*/) -> void {}
  virtual auto on_pin_disconnected(input_pin& /*this_pin*/) -> void {}
  virtual auto on_pin_disconnected(output_pin& /*this_pin*/) -> void {}
  virtual auto type_name() const -> std::string_view = 0;

};
//==============================================================================
}  // namespace base
//==============================================================================
template <typename T>
struct node_serializer {
  //----------------------------------------------------------------------------
  auto serialize(T const& t) const -> toml::table {
    toml::table serialized_node;
    reflection::for_each(t, [&serialized_node](auto const& name,
                                               auto const& var) {
      using var_t = std::decay_t<decltype(var)>;
      if constexpr (std::is_same_v<std::string, var_t>) {
        serialized_node.insert(name, var);
      } else if constexpr (std::is_arithmetic_v<var_t>) {
        serialized_node.insert(name, var);
      } else if constexpr (std::is_same_v<std::array<int, 2>, var_t> ||
                           std::is_same_v<std::array<float, 2>, var_t> ||
                           std::is_same_v<std::array<double, 2>, var_t> ||
                           std::is_same_v<vec<int, 2>, var_t> ||
                           std::is_same_v<vec<float, 2>, var_t> ||
                           std::is_same_v<vec<double, 2>, var_t>) {
        serialized_node.insert(name, toml::array{var.at(0), var.at(1)});
      } else if constexpr (std::is_same_v<std::array<int, 3>, var_t> ||
                           std::is_same_v<std::array<float, 3>, var_t> ||
                           std::is_same_v<std::array<double, 3>, var_t> ||
                           std::is_same_v<vec<int, 3>, var_t> ||
                           std::is_same_v<vec<float, 3>, var_t> ||
                           std::is_same_v<vec<double, 3>, var_t>) {
        serialized_node.insert(name,
                               toml::array{var.at(0), var.at(1), var.at(2)});
      } else if constexpr (std::is_same_v<std::array<int, 4>, var_t> ||
                           std::is_same_v<std::array<float, 4>, var_t> ||
                           std::is_same_v<std::array<double, 4>, var_t> ||
                           std::is_same_v<vec<int, 4>, var_t> ||
                           std::is_same_v<vec<float, 4>, var_t> ||
                           std::is_same_v<vec<double, 4>, var_t>) {
        serialized_node.insert(
            name, toml::array{var.at(0), var.at(1), var.at(2), var.at(3)});
      } else if constexpr (
          std::is_same_v<int[2], std::remove_cv_t<
                                     std::remove_reference_t<decltype(var)>>> ||
          std::is_same_v<
              float[2],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          std::is_same_v<
              double[2],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        serialized_node.insert(name, toml::array{var[0], var[1]});
      } else if constexpr (
          std::is_same_v<int[3], std::remove_cv_t<
                                     std::remove_reference_t<decltype(var)>>> ||
          std::is_same_v<
              float[3],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          std::is_same_v<
              double[3],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        serialized_node.insert(name, toml::array{var[0], var[1], var[2]});
      } else if constexpr (
          std::is_same_v<int[4], std::remove_cv_t<
                                     std::remove_reference_t<decltype(var)>>> ||
          std::is_same_v<
              float[4],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          std::is_same_v<
              double[4],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        serialized_node.insert(name,
                               toml::array{var[0], var[1], var[2], var[3]});
      }
    });
    return serialized_node;
  }
  //----------------------------------------------------------------------------
  auto deserialize(T& t, toml::table const& serialized_node) -> void {
    reflection::for_each(t, [&serialized_node](auto const& name, auto& var) {
      using var_t = std::decay_t<decltype(var)>;
      if constexpr (std::is_same_v<std::string, var_t>) {
        var = serialized_node[name].as_string()->get();
      } else if constexpr (std::is_same_v<bool, var_t>) {
        var = serialized_node[name].as_boolean()->get();
      } else if constexpr (std::is_integral_v<var_t>) {
        var = serialized_node[name].as_integer()->get();
      } else if constexpr (std::is_floating_point_v<var_t>) {
        var = serialized_node[name].as_floating_point()->get();
      } else if constexpr (std::is_same_v<std::array<int, 2>, var_t> ||
                           std::is_same_v<vec<int, 2>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_integer()->get();
        var.at(1) = arr[1].as_integer()->get();
      } else if constexpr (std::is_same_v<std::array<int, 3>, var_t> ||
                           std::is_same_v<vec<int, 3>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_integer()->get();
        var.at(1) = arr[1].as_integer()->get();
        var.at(2) = arr[2].as_integer()->get();
      } else if constexpr (std::is_same_v<std::array<int, 4>, var_t> ||
                           std::is_same_v<vec<int, 4>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_integer()->get();
        var.at(1) = arr[1].as_integer()->get();
        var.at(2) = arr[2].as_integer()->get();
        var.at(3) = arr[3].as_integer()->get();

      } else if constexpr (std::is_same_v<std::array<float, 2>, var_t> ||
                           std::is_same_v<std::array<double, 2>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_floating_point()->get();
        var[1] = arr[1].as_floating_point()->get();
      } else if constexpr (std::is_same_v<vec<float, 2>, var_t> ||
                           std::is_same_v<vec<double, 2>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var(0) = arr[0].as_floating_point()->get();
        var(1) = arr[1].as_floating_point()->get();
      } else if constexpr (std::is_same_v<std::array<float, 3>, var_t> ||
                           std::is_same_v<std::array<double, 3>, var_t> ||
                           std::is_same_v<vec<float, 3>, var_t> ||
                           std::is_same_v<vec<double, 3>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_floating_point()->get();
        var.at(1) = arr[1].as_floating_point()->get();
        var.at(2) = arr[2].as_floating_point()->get();
      } else if constexpr (std::is_same_v<std::array<float, 4>, var_t> ||
                           std::is_same_v<std::array<double, 4>, var_t> ||
                           std::is_same_v<vec<float, 4>, var_t> ||
                           std::is_same_v<vec<double, 4>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_floating_point()->get();
        var.at(1) = arr[1].as_floating_point()->get();
        var.at(2) = arr[2].as_floating_point()->get();
        var.at(3) = arr[3].as_floating_point()->get();
      } else if constexpr (std::is_same_v<
                               int[2], std::remove_cv_t<std::remove_reference_t<
                                           decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_integer()->get();
        var[1] = arr[1].as_integer()->get();
      } else if constexpr (std::is_same_v<
                               int[3], std::remove_cv_t<std::remove_reference_t<
                                           decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_integer()->get();
        var[1] = arr[1].as_integer()->get();
        var[2] = arr[2].as_integer()->get();
      } else if constexpr (std::is_same_v<
                               int[4], std::remove_cv_t<std::remove_reference_t<
                                           decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_integer()->get();
        var[1] = arr[1].as_integer()->get();
        var[2] = arr[2].as_integer()->get();
        var[3] = arr[3].as_integer()->get();

      } else if constexpr (
          std::is_same_v<
              float[2],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          std::is_same_v<
              double[2],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at[0] = arr[0].as_floating_point()->get();
        var.at[1] = arr[1].as_floating_point()->get();
      } else if constexpr (
          std::is_same_v<
              float[3],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          std::is_same_v<
              double[3],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_floating_point()->get();
        var[1] = arr[1].as_floating_point()->get();
        var[2] = arr[2].as_floating_point()->get();
      } else if constexpr (
          std::is_same_v<
              float[4],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          std::is_same_v<
              double[4],
              std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_floating_point()->get();
        var[1] = arr[1].as_floating_point()->get();
        var[2] = arr[2].as_floating_point()->get();
        var[3] = arr[3].as_floating_point()->get();
      }
    });
  }
  //----------------------------------------------------------------------------
  auto draw_properties(T& t) -> bool {
    bool changed = false;
    reflection::for_each(t, [&changed](auto const& name, auto& var) {
      using var_t = std::decay_t<decltype(var)>;
      // float
      if constexpr (std::is_same_v<std::string, var_t>) {
        changed |= ImGui::InputText(name, &var);
      } else if constexpr (std::is_same_v<float, var_t>) {
        changed |= ImGui::DragFloat(name, &var, 0.1f);
      } else if constexpr (std::is_same_v<std::array<float, 2>, var_t>) {
        changed |= ImGui::DragFloat2(name, var.data(), 0.1f);
      } else if constexpr (std::is_same_v<std::array<float, 3>, var_t>) {
        changed |= ImGui::DragFloat3(name, var.data(), 0.1f);
      } else if constexpr (std::is_same_v<std::array<float, 4>, var_t>) {
        changed |= ImGui::DragFloat4(name, var.data(), 0.1f);
      } else if constexpr (std::is_same_v<vec<float, 2>, var_t>) {
        changed |= ImGui::DragFloat2(name, var.data_ptr(), 0.1f);
      } else if constexpr (std::is_same_v<vec<float, 3>, var_t>) {
        changed |= ImGui::DragFloat3(name, var.data_ptr(), 0.1f);
      } else if constexpr (std::is_same_v<vec<float, 4>, var_t>) {
        changed |= ImGui::DragFloat4(name, var.data_ptr(), 0.1f);

        // double
      } else if constexpr (std::is_same_v<double, var_t>) {
        changed |= ImGui::DragDouble(name, &var, 0.1);
      } else if constexpr (std::is_same_v<std::array<double, 2>, var_t>) {
        changed |= ImGui::DragDouble2(name, var.data(), 0.1);
      } else if constexpr (std::is_same_v<std::array<double, 3>, var_t>) {
        changed |= ImGui::DragDouble3(name, var.data(), 0.1);
      } else if constexpr (std::is_same_v<std::array<double, 4>, var_t>) {
        changed |= ImGui::DragDouble4(name, var.data(), 0.1);
      } else if constexpr (std::is_same_v<vec<double, 2>, var_t>) {
        changed |= ImGui::DragDouble2(name, var.data_ptr(), 0.1);
      } else if constexpr (std::is_same_v<vec<double, 3>, var_t>) {
        changed |= ImGui::DragDouble3(name, var.data_ptr(), 0.1);
      } else if constexpr (std::is_same_v<vec<double, 4>, var_t>) {
        changed |= ImGui::DragDouble4(name, var.data_ptr(), 0.1);

        // int
      } else if constexpr (std::is_same_v<int, var_t>) {
        changed |= ImGui::DragInt(name, &var, 1);
      } else if constexpr (std::is_same_v<std::array<int, 2>, var_t>) {
        changed |= ImGui::DragInt2(name, var.data(), 1);
      } else if constexpr (std::is_same_v<std::array<int, 3>, var_t>) {
        changed |= ImGui::DragInt3(name, var.data(), 1);
      } else if constexpr (std::is_same_v<std::array<int, 4>, var_t>) {
        changed |= ImGui::DragInt4(name, var.data(), 1);
      } else if constexpr (std::is_same_v<vec<int, 2>, var_t>) {
        changed |= ImGui::DragInt2(name, var.data_ptr(), 1);
      } else if constexpr (std::is_same_v<vec<int, 3>, var_t>) {
        changed |= ImGui::DragInt3(name, var.data_ptr(), 1);
      } else if constexpr (std::is_same_v<vec<int, 4>, var_t>) {
        changed |= ImGui::DragInt4(name, var.data_ptr(), 1);
      }
    });
    return changed;
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
  auto draw_properties() -> bool override {
    return node_serializer<Child>::draw_properties(*dynamic_cast<Child*>(this));
  }
  //----------------------------------------------------------------------------
  auto type_name() const -> std::string_view override final {
    return node_serializer<Child>::type_name();
  }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::ui
//==============================================================================
struct registration_name_t {
  using F = std::string_view (*)();
  F f;
};
//------------------------------------------------------------------------------
struct registration_factory_t {
  using F = tatooine::flowexplorer::ui::base::
      node* (*)(tatooine::flowexplorer::scene&, std::string_view const&);
  F f;
};
//==============================================================================
#define TATOOINE_FLOWEXPLORER_REGISTER_NAME(registered_function_, sec)         \
  static registration_name_t ptr_##registered_function_                        \
      __attribute((used, section(#sec))) = {                                   \
          .f = registered_function_,                                           \
  }
//------------------------------------------------------------------------------
#define TATOOINE_FLOWEXPLORER_REGISTER_FACTORY(registered_function_, sec)      \
  static registration_factory_t ptr_##registered_function_                     \
      __attribute((used, section(#sec))) = {                                   \
          .f = registered_function_,                                           \
  }
//------------------------------------------------------------------------------
#define TATOOINE_FLOWEXPLORER_REGISTER_NODE(type, ...)                         \
  TATOOINE_MAKE_ADT_REFLECTABLE(type, ##__VA_ARGS__)                           \
  namespace tatooine::flowexplorer::registration::type {                       \
  static auto factory(::tatooine::flowexplorer::scene& s,                      \
                      std::string_view const&          node_type_name)         \
      -> ::tatooine::flowexplorer::ui::base::node* {                           \
    if (node_type_name == #type) {                                             \
      return s.nodes().emplace_back(new ::type{s}).get();                      \
    }                                                                          \
    return nullptr;                                                            \
  }                                                                            \
  static constexpr auto name() -> std::string_view { return #type; }           \
  TATOOINE_FLOWEXPLORER_REGISTER_FACTORY(factory, factory_);                   \
  TATOOINE_FLOWEXPLORER_REGISTER_NAME(name, name_);                            \
  }
//------------------------------------------------------------------------------
#define TATOOINE_FLOWEXPLORER_REGISTER_RENDERABLE(type, ...)                   \
  TATOOINE_MAKE_ADT_REFLECTABLE(type, ##__VA_ARGS__)                           \
  namespace tatooine::flowexplorer::registration::type {                       \
  static auto factory(::tatooine::flowexplorer::scene& s,                      \
                      std::string_view const&          node_type_name)         \
      -> ::tatooine::flowexplorer::ui::base::node* {                           \
    if (node_type_name == #type) {                                             \
      return s.renderables().emplace_back(new ::type{s}).get();                \
    }                                                                          \
    return nullptr;                                                            \
  }                                                                            \
  static constexpr auto name() -> std::string_view { return #type; }           \
  TATOOINE_FLOWEXPLORER_REGISTER_FACTORY(factory, factory_);                   \
  TATOOINE_FLOWEXPLORER_REGISTER_NAME(name, name_);                            \
  }
//------------------------------------------------------------------------------
#define iterate_registered_factories(elem)                                     \
  for (registration_factory_t* elem = &__start_factory_;                       \
       elem != &__stop_factory_; ++elem)
//------------------------------------------------------------------------------
#define iterate_registered_names(elem)                                         \
  for (registration_name_t* elem = &__start_name_; elem != &__stop_name_;      \
       ++elem)
//------------------------------------------------------------------------------
extern registration_factory_t __start_factory_;
extern registration_factory_t __stop_factory_;
extern registration_name_t    __start_name_;
extern registration_name_t    __stop_name_;
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
auto insert_registered_element(scene& s, std::string_view const& name)
    -> ui::base::node*;
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
