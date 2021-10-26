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
struct is_reflectable_impl : std::integral_constant<bool, reflector<T>::value> {
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_reflectable = is_reflectable_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
concept reflectable = is_reflectable<std::decay_t<T>>;
//==============================================================================
// User-interface
//==============================================================================
/// Return number of fields in a reflector struct
template <typename T>
constexpr auto num_members() -> std::size_t {
  return reflector<std::decay_t<T>>::num_members();
}
//------------------------------------------------------------------------------
template <typename T>
constexpr auto num_members(T &&) -> std::size_t {
  return num_members<T>();
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
  static_assert(is_reflectable<T>);
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
  template <reflectable Reflectable>                                           \
  static constexpr auto get(                                                   \
      index_t<static_cast<std::size_t>(field_indices::NAME)>, Reflectable &&t) \
      ->decltype(auto) {                                                       \
    return t.ACCESSOR;                                                         \
  }                                                                            \
  static constexpr auto name(                                                  \
      index_t<static_cast<std::size_t>(field_indices::NAME)>)                  \
      ->std::string_view {                                                     \
    return #NAME;                                                              \
  }
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_FIELD_INDICES(...)                                 \
  enum class field_indices : std::size_t { __VA_ARGS__, count___ };            \
//==============================================================================
/// Takes a type and a set of public member variables.
#define TATOOINE_MAKE_SIMPLE_TYPE_REFLECTABLE(TYPE, ...)                       \
  TATOOINE_MAKE_ADT_REFLECTABLE(                                               \
      TYPE, TATOOINE_PP_MAP(TATOOINE_REFLECTION_INSERT_MEMBER, __VA_ARGS__))
//------------------------------------------------------------------------------
#define TATOOINE_MAKE_ADT_REFLECTABLE(TYPE, ...)                               \
  template <>                                                                  \
  TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(TYPE, __VA_ARGS__)
//------------------------------------------------------------------------------
#define TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(TYPE, ...)                      \
  struct reflector<TATOOINE_PP_PASS_ARGS(TYPE)> {                               \
    template <std::size_t I>                                                    \
    using index_t                     = std::integral_constant<std::size_t, I>; \
    using reflected_type              = TATOOINE_PP_PASS_ARGS(TYPE);            \
    static constexpr const bool value = true;                                   \
                                                                                \
    TATOOINE_REFLECTION_FIELD_INDICES(TATOOINE_PP_ODDS(__VA_ARGS__))            \
    static constexpr auto num_members() {                                       \
      return static_cast<std::size_t>(field_indices::count___);                 \
    }                                                                           \
    static constexpr auto name() -> std::string_view {                          \
      return TATOOINE_PP_TO_STRING(TYPE);                                       \
    }                                                                           \
                                                                                \
    TATOOINE_PP_MAP2(TATOOINE_REFLECTION_MAKE_GETTERS, ##__VA_ARGS__)           \
                                                                                \
    template <reflectable Reflectable, typename V>                              \
    constexpr static auto for_each([[maybe_unused]] V &&          v,            \
                                   [[maybe_unused]] Reflectable &&t) -> void {  \
      for_each(std::forward<V>(v), std::forward<Reflectable>(t),                \
               std::make_index_sequence<num_members()>{});                      \
    }                                                                           \
    template <reflectable Reflectable, typename V, std::size_t... Is>           \
    constexpr static auto for_each([[maybe_unused]] V &&          v,            \
                                   [[maybe_unused]] Reflectable &&t,            \
                                   std::index_sequence<Is...>) -> void {        \
      (v(name(index_t<Is>{}), get(index_t<Is>{}, t)), ...);                     \
    }                                                                           \
  };
//==============================================================================
}  // namespace tatooine::reflection
//==============================================================================
#endif
