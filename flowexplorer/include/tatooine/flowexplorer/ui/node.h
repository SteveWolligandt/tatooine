#ifndef TATOOINE_FLOWEXPLORER_UI_NODE_H
#define TATOOINE_FLOWEXPLORER_UI_NODE_H
//==============================================================================
#include <imgui-node-editor/imgui_node_editor.h>
#include <tatooine/demangling.h>
#include <tatooine/flowexplorer/serializable.h>
#include <tatooine/flowexplorer/toggleable.h>
#include <tatooine/flowexplorer/ui/pin.h>
#include <tatooine/flowexplorer/uuid_holder.h>
#include <tatooine/reflection.h>
#include <tatooine/vec.h>
#include <tatooine/gl/imgui.h>
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
struct node : uuid_holder<ax::NodeEditor::NodeId>, serializable, toggleable {
 private:
  std::string                                           m_title;
  flowexplorer::scene*                                  m_scene;
  std::vector<std::unique_ptr<input_pin>>               m_input_pins;
  std::vector<std::unique_ptr<output_pin>>              m_output_pins;
  std::vector<std::unique_ptr<input_pin_property_link>> m_property_links;
  std::unique_ptr<output_pin>                           m_self_pin = nullptr;

 public:
  node(flowexplorer::scene& s);
  node(std::string const& title, flowexplorer::scene& s);
  template <typename T>
  node(flowexplorer::scene& s, T& ref) : node{s} {
    m_self_pin = make_output_pin(*this, "", ref);
  }
  //------------------------------------------------------------------------------
  template <typename T>
  node(std::string const& title, flowexplorer::scene& s, T& ref)
      : node{title, s} {
    m_self_pin = make_output_pin(*this, "", ref);
  }
  //------------------------------------------------------------------------------
  virtual ~node() = default;
  //============================================================================
  template <typename... Ts>
  auto insert_input_pin(std::string const& title,
                        icon_type const    t = icon_type::flow) -> auto& {
    m_input_pins.push_back(make_input_pin<Ts...>(*this, title, t));
    return *m_input_pins.back();
  }
  //----------------------------------------------------------------------------
  template <typename Prop>
  auto insert_input_pin_property_link(input_pin& pin, Prop& prop) -> auto& {
    pin.set_icon_type(icon_type::grid);
    m_property_links.push_back(make_input_pin_property_link<Prop>(pin, prop));
    return *m_property_links.back();
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto insert_output_pin(std::string const& title, T& ref,
                        icon_type const    t = icon_type::flow) -> auto& {
    m_output_pins.push_back(make_output_pin(*this, title, ref, t));
    return *m_output_pins.back();
  }
  //----------------------------------------------------------------------------
  auto title() const -> auto const& { return m_title; }
  auto title() -> auto& { return m_title; }
  //----------------------------------------------------------------------------
  auto scene() const -> auto const& { return *m_scene; }
  auto scene() -> auto& { return *m_scene; }
  //----------------------------------------------------------------------------
  auto set_title(std::string const& title) { m_title = title; }
  //----------------------------------------------------------------------------
  auto has_self_pin() const -> bool { return m_self_pin != nullptr; }
  //----------------------------------------------------------------------------
  auto self_pin() const -> auto const& { return *m_self_pin; }
  auto self_pin() -> auto& { return *m_self_pin; }
  //----------------------------------------------------------------------------
  auto input_pins() const -> auto const& { return m_input_pins; }
  auto input_pins() -> auto& { return m_input_pins; }
  //----------------------------------------------------------------------------
  auto output_pins() const -> auto const& { return m_output_pins; }
  auto output_pins() -> auto& { return m_output_pins; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto draw_node() -> void;
  //----------------------------------------------------------------------------
  auto node_position() const -> ImVec2;
  //----------------------------------------------------------------------------
  auto notify_property_changed(bool const notify_self = true) -> void;
  //----------------------------------------------------------------------------
  auto update_property_links() -> void {
    bool changed = false;
    for (auto& prop_link : m_property_links) {
      changed |= prop_link->update();
    }
    if (changed) {
      notify_property_changed();
    }
  }
  //----------------------------------------------------------------------------
  // virtual methods
  //----------------------------------------------------------------------------
  virtual auto draw_properties() -> bool = 0;
  virtual auto on_property_changed() -> void {}
  virtual auto on_title_changed(std::string const& /*old_title*/) -> void {}
  virtual auto on_pin_connected(input_pin& /*this_pin*/,
                                output_pin& /*other_pin*/) -> void {}
  virtual auto on_pin_connected(output_pin& /*this_pin*/,
                                input_pin& /*other_pin*/) -> void {}
  virtual auto on_pin_disconnected(input_pin& /*this_pin*/) -> void {}
  virtual auto on_pin_disconnected(output_pin& /*this_pin*/) -> void {}
  virtual auto type_name() const -> std::string_view = 0;
  virtual auto update(std::chrono::duration<double> const& /*dt*/) -> void {}
  virtual auto on_path_selected(std::string const& /*path*/) -> void {}
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
      if constexpr (is_same<size_t, var_t>) {
        serialized_node.insert(name, (int)var);
      } else if constexpr (is_same<std::string, var_t>) {
        serialized_node.insert(name, var);
      } else if constexpr (std::is_arithmetic_v<var_t>) {
        serialized_node.insert(name, var);
      } else if constexpr (is_same<std::array<int, 2>, var_t> ||
                           is_same<std::array<float, 2>, var_t> ||
                           is_same<std::array<double, 2>, var_t> ||
                           is_same<vec<int, 2>, var_t> ||
                           is_same<vec<float, 2>, var_t> ||
                           is_same<vec<double, 2>, var_t>) {
        serialized_node.insert(name, toml::array{var.at(0), var.at(1)});
      } else if constexpr (is_same<std::array<int, 3>, var_t> ||
                           is_same<std::array<float, 3>, var_t> ||
                           is_same<std::array<double, 3>, var_t> ||
                           is_same<vec<int, 3>, var_t> ||
                           is_same<vec<float, 3>, var_t> ||
                           is_same<vec<double, 3>, var_t>) {
        serialized_node.insert(name,
                               toml::array{var.at(0), var.at(1), var.at(2)});
      } else if constexpr (is_same<std::array<int, 4>, var_t> ||
                           is_same<std::array<float, 4>, var_t> ||
                           is_same<std::array<double, 4>, var_t> ||
                           is_same<vec<int, 4>, var_t> ||
                           is_same<vec<float, 4>, var_t> ||
                           is_same<vec<double, 4>, var_t>) {
        serialized_node.insert(
            name, toml::array{var.at(0), var.at(1), var.at(2), var.at(3)});
      } else if constexpr (
          is_same<int[2],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          is_same<float[2],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          is_same<double[2],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        serialized_node.insert(name, toml::array{var[0], var[1]});
      } else if constexpr (
          is_same<int[3],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          is_same<float[3],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          is_same<double[3],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        serialized_node.insert(name, toml::array{var[0], var[1], var[2]});
      } else if constexpr (
          is_same<int[4],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          is_same<float[4],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          is_same<double[4],
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
      if constexpr (is_same<size_t, var_t>) {
        var = (size_t)serialized_node[name].as_integer()->get();
      } else if constexpr (is_same<std::string, var_t>) {
        var = serialized_node[name].as_string()->get();
      } else if constexpr (is_same<bool, var_t>) {
        var = serialized_node[name].as_boolean()->get();
      } else if constexpr (std::is_integral_v<var_t>) {
        var = serialized_node[name].as_integer()->get();
      } else if constexpr (std::is_floating_point_v<var_t>) {
        var = serialized_node[name].as_floating_point()->get();
      } else if constexpr (is_same<std::array<int, 2>, var_t> ||
                           is_same<vec<int, 2>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_integer()->get();
        var.at(1) = arr[1].as_integer()->get();
      } else if constexpr (is_same<std::array<int, 3>, var_t> ||
                           is_same<vec<int, 3>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_integer()->get();
        var.at(1) = arr[1].as_integer()->get();
        var.at(2) = arr[2].as_integer()->get();
      } else if constexpr (is_same<std::array<int, 4>, var_t> ||
                           is_same<vec<int, 4>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_integer()->get();
        var.at(1) = arr[1].as_integer()->get();
        var.at(2) = arr[2].as_integer()->get();
        var.at(3) = arr[3].as_integer()->get();

      } else if constexpr (is_same<std::array<float, 2>, var_t> ||
                           is_same<std::array<double, 2>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_floating_point()->get();
        var[1] = arr[1].as_floating_point()->get();
      } else if constexpr (is_same<vec<float, 2>, var_t> ||
                           is_same<vec<double, 2>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var(0) = arr[0].as_floating_point()->get();
        var(1) = arr[1].as_floating_point()->get();
      } else if constexpr (is_same<std::array<float, 3>, var_t> ||
                           is_same<std::array<double, 3>, var_t> ||
                           is_same<vec<float, 3>, var_t> ||
                           is_same<vec<double, 3>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_floating_point()->get();
        var.at(1) = arr[1].as_floating_point()->get();
        var.at(2) = arr[2].as_floating_point()->get();
      } else if constexpr (is_same<std::array<float, 4>, var_t> ||
                           is_same<std::array<double, 4>, var_t> ||
                           is_same<vec<float, 4>, var_t> ||
                           is_same<vec<double, 4>, var_t>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at(0) = arr[0].as_floating_point()->get();
        var.at(1) = arr[1].as_floating_point()->get();
        var.at(2) = arr[2].as_floating_point()->get();
        var.at(3) = arr[3].as_floating_point()->get();
      } else if constexpr (is_same<int[2],
                                   std::remove_cv_t<std::remove_reference_t<
                                       decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_integer()->get();
        var[1] = arr[1].as_integer()->get();
      } else if constexpr (is_same<int[3],
                                   std::remove_cv_t<std::remove_reference_t<
                                       decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_integer()->get();
        var[1] = arr[1].as_integer()->get();
        var[2] = arr[2].as_integer()->get();
      } else if constexpr (is_same<int[4],
                                   std::remove_cv_t<std::remove_reference_t<
                                       decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_integer()->get();
        var[1] = arr[1].as_integer()->get();
        var[2] = arr[2].as_integer()->get();
        var[3] = arr[3].as_integer()->get();

      } else if constexpr (
          is_same<float[2],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          is_same<double[2],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var.at[0] = arr[0].as_floating_point()->get();
        var.at[1] = arr[1].as_floating_point()->get();
      } else if constexpr (
          is_same<float[3],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          is_same<double[3],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>>) {
        auto const& arr = *serialized_node[name].as_array();

        var[0] = arr[0].as_floating_point()->get();
        var[1] = arr[1].as_floating_point()->get();
        var[2] = arr[2].as_floating_point()->get();
      } else if constexpr (
          is_same<float[4],
                  std::remove_cv_t<std::remove_reference_t<decltype(var)>>> ||
          is_same<double[4],
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
      if constexpr (is_same<std::string, var_t>) {
        changed |= ImGui::InputText(name, &var);

      } else if constexpr (is_same<size_t, var_t>) {
        changed |= ImGui::DragSizeT(name, &var);
        
        // float
      } else if constexpr (is_same<float, var_t>) {
        changed |= ImGui::DragFloat(name, &var, 0.1f);
      } else if constexpr (is_same<std::array<float, 2>, var_t>) {
        changed |= ImGui::DragFloat2(name, var.data(), 0.1f);

      } else if constexpr (is_same<std::array<float, 3>, var_t>) {
        changed |= ImGui::DragFloat3(name, var.data(), 0.1f);
      } else if constexpr (is_same<std::array<float, 4>, var_t>) {
        changed |= ImGui::DragFloat4(name, var.data(), 0.1f);
      } else if constexpr (is_same<vec<float, 2>, var_t>) {
        changed |= ImGui::DragFloat2(name, var.data(), 0.1f);
      } else if constexpr (is_same<vec<float, 3>, var_t>) {
        changed |= ImGui::DragFloat3(name, var.data(), 0.1f);
      } else if constexpr (is_same<vec<float, 4>, var_t>) {
        changed |= ImGui::DragFloat4(name, var.data(), 0.1f);

        // double
      } else if constexpr (is_same<double, var_t>) {
        changed |= ImGui::DragDouble(name, &var, 0.1);
      } else if constexpr (is_same<std::array<double, 2>, var_t>) {
        changed |= ImGui::DragDouble2(name, var.data(), 0.1);
      } else if constexpr (is_same<std::array<double, 3>, var_t>) {
        changed |= ImGui::DragDouble3(name, var.data(), 0.1);
      } else if constexpr (is_same<std::array<double, 4>, var_t>) {
        changed |= ImGui::DragDouble4(name, var.data(), 0.1);
      } else if constexpr (is_same<vec<double, 2>, var_t>) {
        changed |= ImGui::DragDouble2(name, var.data(), 0.1);
      } else if constexpr (is_same<vec<double, 3>, var_t>) {
        changed |= ImGui::DragDouble3(name, var.data(), 0.1);
      } else if constexpr (is_same<vec<double, 4>, var_t>) {
        changed |= ImGui::DragDouble4(name, var.data(), 0.1);

        // int
      } else if constexpr (is_same<int, var_t>) {
        changed |= ImGui::DragInt(name, &var, 1);
      } else if constexpr (is_same<std::array<int, 2>, var_t>) {
        changed |= ImGui::DragInt2(name, var.data(), 1);
      } else if constexpr (is_same<std::array<int, 3>, var_t>) {
        changed |= ImGui::DragInt3(name, var.data(), 1);
      } else if constexpr (is_same<std::array<int, 4>, var_t>) {
        changed |= ImGui::DragInt4(name, var.data(), 1);
      } else if constexpr (is_same<vec<int, 2>, var_t>) {
        changed |= ImGui::DragInt2(name, var.data(), 1);
      } else if constexpr (is_same<vec<int, 3>, var_t>) {
        changed |= ImGui::DragInt3(name, var.data(), 1);
      } else if constexpr (is_same<vec<int, 4>, var_t>) {
        changed |= ImGui::DragInt4(name, var.data(), 1);
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
  auto deserialize(toml::table const& serialized_node) -> void override {
    node_serializer<Child>::deserialize(*dynamic_cast<Child*>(this),
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

