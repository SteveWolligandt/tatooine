#ifndef TATOOINE_REFLECTION_H
#define TATOOINE_REFLECTION_H
//==============================================================================
// Provides a facility to declare a type as "reflector" and apply a
// reflection_visitor to it. The list of members is a compile-time data
// structure, and there is no run-time overhead.
//==============================================================================
#include <tatooine/preprocessor.h>

#include <concepts>
#include <string_view>
#include <type_traits>
#include <utility>
//==============================================================================
namespace tatooine::reflection {
//==============================================================================
// Primary template which is specialized to register a type
template <typename T>
struct reflector {
  static constexpr const bool value = false;
};
//==============================================================================
template <typename T>
struct is_reflectable : std::integral_constant<bool, reflector<T>::value> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_reflectable_v = is_reflectable<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
concept reflectable = is_reflectable_v<std::decay_t<T>>;
//==============================================================================
// User-interface
//==============================================================================
/// Return number of fields in a reflector struct
template <typename T>
constexpr auto num_fields() -> std::size_t {
  return reflector<std::decay_t<T>>::num_fields();
}
//------------------------------------------------------------------------------
template <typename T>
constexpr auto num_fields(T &&) -> std::size_t {
  return num_fields<T>();
}
//------------------------------------------------------------------------------
/// Iterate over each registered member
template <reflectable T, typename V>
constexpr auto for_each(T &&t, V &&v) {
  reflector<std::decay_t<T>>::for_each(std::forward<V>(v), std::forward<T>(t));
}
//------------------------------------------------------------------------------
// Get value by index (like std::get for tuples)
template <std::size_t idx, reflectable T>
constexpr auto get(T &&t) {
  return reflector<std::decay_t<T>>::get(
      std::integral_constant<std::size_t, idx>{}, std::forward<T>(t));
}
//------------------------------------------------------------------------------
// Get name of field, by index
template <std::size_t idx, reflectable T>
constexpr auto name() {
  return reflector<std::decay_t<T>>::name(
      std::integral_constant<std::size_t, idx>{});
}
//------------------------------------------------------------------------------
template <std::size_t idx, typename T>
constexpr auto name(T &&) -> decltype(name<idx, T>()) {
  return name<idx, T>();
}
//------------------------------------------------------------------------------
// Get name of structure
template <reflectable T>
constexpr auto name() {
  return reflector<std::decay_t<T>>::name();
}
//------------------------------------------------------------------------------
template <reflectable T>
constexpr auto name(T &&) {
  return name<T>();
}
//------------------------------------------------------------------------------
template <std::size_t I, typename T>
struct get_type_impl {
  static_assert(is_reflectable_v<T>);
  using type = decltype(reflector<std::decay_t<T>>::get(
      std::integral_constant<std::size_t, I>{}, std::declval<T>()));
};
//------------------------------------------------------------------------------
template <std::size_t I, typename T>
using get_type = typename get_type_impl<I, T>::type;
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_INSERT_MEMBER(MEMBER)                              \
  MEMBER TATOOINE_PP_COMMA() MEMBER
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_INSERT_GETTER(GETTER)                              \
  GETTER TATOOINE_PP_COMMA() GETTER()
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_INSERT_METHOD(NAME, ACCESSOR)                      \
  NAME TATOOINE_PP_COMMA() ACCESSOR
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_NAME(NAME, ACCESSOR) NAME
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_ACCESSOR(NAME, ACCESSOR) ACCESSOR
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_MEMBER_HELPER(NAME, ACCESSOR) v(#NAME, t.ACCESSOR);
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_MAKE_GETTERS(NAME, ACCESSOR)                       \
  template <typename T>                                                        \
  static constexpr auto get(                                                   \
      index_t<static_cast<std::size_t>(field_indices::NAME)>, T &&t)           \
      ->decltype(auto) {                                                       \
    return t.ACCESSOR;                                                         \
  }                                                                            \
  static constexpr auto name(                                                  \
      index_t<static_cast<std::size_t>(field_indices::NAME)>)                  \
      ->std::string_view {                                                     \
    return #NAME;                                                              \
  }
//==============================================================================
#define TATOOINE_MAKE_REFLECTABLE(STRUCT_NAME, ...)                            \
  TATOOINE_MAKE_ADT_REFLECTABLE(                                               \
      STRUCT_NAME,                                                             \
      TATOOINE_PP_MAP(TATOOINE_REFLECTION_INSERT_MEMBER, __VA_ARGS__))
//------------------------------------------------------------------------------
#define TATOOINE_MAKE_ADT_REFLECTABLE(STRUCT_NAME, ...)                         \
  namespace tatooine::reflection {                                              \
  template <>                                                                   \
  struct reflector<STRUCT_NAME> {                                               \
    template <std::size_t I>                                                    \
    using index_t                     = std::integral_constant<std::size_t, I>; \
    using reflected_type              = STRUCT_NAME;                            \
    static constexpr const bool value = true;                                   \
                                                                                \
    enum class field_indices : std::size_t {                                    \
      TATOOINE_PP_ODDS(__VA_ARGS__),                                            \
      count                                                                     \
    };                                                                          \
    static constexpr auto num_fields() {                                        \
      return static_cast<std::size_t>(field_indices::count);                    \
    }                                                                           \
    static constexpr auto name() -> std::string_view { return #STRUCT_NAME; }   \
                                                                                \
    TATOOINE_PP_MAP2(TATOOINE_REFLECTION_MAKE_GETTERS, ##__VA_ARGS__)           \
                                                                                \
    template <reflectable T, typename V>                                        \
    constexpr static auto for_each([[maybe_unused]] V &&v,                      \
                                   [[maybe_unused]] T &&t) -> void {            \
      for_each(std::forward<V>(v), std::forward<T>(t),                          \
               std::make_index_sequence<num_fields()>{});                       \
    }                                                                           \
    template <reflectable T, typename V, std::size_t... Is>                     \
    constexpr static auto for_each([[maybe_unused]] V &&v,                      \
                                   [[maybe_unused]] T &&t,                      \
                                   std::index_sequence<Is...>) -> void {        \
      (v(name(index_t<Is>{}), get(index_t<Is>{}, t)), ...);                     \
    }                                                                           \
  };                                                                            \
  }
//==============================================================================
}  // namespace tatooine::reflection
//==============================================================================
#endif
