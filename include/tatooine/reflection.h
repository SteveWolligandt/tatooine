#ifndef TATOOINE_REFLECTION_H
#define TATOOINE_REFLECTION_H
//==============================================================================
// Provides a facility to declare a type as "reflectable" and apply a
// reflection_visitor to it. The list of members is a compile-time data
// structure, and there is no run-time overhead.
//==============================================================================
#include <concepts>
#include <string_view>
#include <type_traits>
#include <utility>

#include <tatooine/preprocessor.h>
//==============================================================================
namespace tatooine::reflection::detail {
//==============================================================================
// Primary template which is specialized to register a type
template <typename T>
struct reflectable {
  static constexpr const bool value = false;
};
//==============================================================================
}  // namespace tatooine::reflection::detail
//==============================================================================
namespace tatooine::reflection {
//==============================================================================
template <typename T>
struct is_reflectable
    : std::integral_constant<bool, detail::reflectable<T>::value> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_reflectable_v = is_reflectable<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
concept reflectable = is_reflectable_v<std::decay_t<T>>;
//==============================================================================
// User-interface
//==============================================================================
// Return number of fields in a reflectable struct
template <typename T>
constexpr auto field_count() -> std::size_t {
  return detail::reflectable<std::decay_t<T>>::field_count;
}
//------------------------------------------------------------------------------
template <typename T>
constexpr auto field_count(T &&) -> std::size_t {
  return field_count<T>();
}
//------------------------------------------------------------------------------
/// Iterate over each registered member
template <reflectable T, typename V>
constexpr auto for_each(T &&t, V &&v) {
  detail::reflectable<std::decay_t<T>>::apply(std::forward<V>(v),
                                              std::forward<T>(t));
}
//------------------------------------------------------------------------------
// Get value by index (like std::get for tuples)
template <int idx, reflectable T>
constexpr auto get(T &&t) {
  return detail::reflectable<std::decay_t<T>>::get_value(
      std::integral_constant<int, idx>{}, std::forward<T>(t));
}
//------------------------------------------------------------------------------
// Get name of field, by index
template <int idx, reflectable T>
constexpr auto name() {
  return detail::reflectable<std::decay_t<T>>::name(
      std::integral_constant<int, idx>{});
}
//------------------------------------------------------------------------------
template <int idx, typename T>
constexpr auto name(T &&) -> decltype(name<idx, T>()) {
  return name<idx, T>();
}
//------------------------------------------------------------------------------
// Get name of structure
template <reflectable T>
constexpr auto name() {
  return detail::reflectable<std::decay_t<T>>::name();
}
//------------------------------------------------------------------------------
template <reflectable T>
constexpr auto name(T &&) {
  return name<T>();
}
//------------------------------------------------------------------------------
// These macros are used with TATOOINE_PP_MAP
#define TATOOINE_REFLECTION_FIELD_COUNT(MEMBER) +1
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_INSERT_MEMBER(MEMBER) \
  MEMBER TATOOINE_PP_COMMA() MEMBER
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_INSERT_GETTER(GETTER) \
  GETTER TATOOINE_PP_COMMA() GETTER()
//------------------------------------------------------------------------------
#define TATOOINE_REFLECTION_INSERT_METHOD(NAME, ACCESSOR) \
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
  static constexpr auto get_value(                                             \
      std::integral_constant<std::size_t, fields_enum::NAME>, T &&t)           \
      ->decltype(auto) {                                                       \
    return t.ACCESSOR;                                                         \
  }                                                                            \
                                                                               \
  static constexpr auto name(                                              \
      std::integral_constant<std::size_t, fields_enum::NAME>)                  \
      ->std::string_view {                                                     \
    return #NAME;                                                              \
  }

//==============================================================================
#define TATOOINE_MAKE_REFLECTABLE(STRUCT_NAME, ...) \
  TATOOINE_MAKE_ADT_REFLECTABLE(                    \
      STRUCT_NAME,                                  \
      TATOOINE_PP_MAP(TATOOINE_REFLECTION_INSERT_MEMBER, __VA_ARGS__))
//------------------------------------------------------------------------------
#define TATOOINE_MAKE_ADT_REFLECTABLE(STRUCT_NAME, ...)                        \
  namespace tatooine::reflection::detail {                                     \
  template <>                                                                  \
  struct reflectable<STRUCT_NAME> {                                            \
    using reflected_type              = STRUCT_NAME;                           \
    static constexpr const bool value = true;                                  \
                                                                               \
    struct fields_enum {                                                       \
      enum index { TATOOINE_PP_ODDS(__VA_ARGS__) };                            \
    };                                                                         \
                                                                               \
    static constexpr auto name() -> std::string_view { return #STRUCT_NAME; }  \
                                                                               \
    TATOOINE_PP_MAP2(TATOOINE_REFLECTION_MAKE_GETTERS, ##__VA_ARGS__)          \
                                                                               \
    template <reflection::reflectable T, typename V>                           \
    constexpr static auto apply([[maybe_unused]] V &&v,                        \
                                [[maybe_unused]] T &&t) -> void {              \
      TATOOINE_PP_MAP2(TATOOINE_REFLECTION_MEMBER_HELPER, ##__VA_ARGS__)       \
    }                                                                          \
  };                                                                           \
  }
//==============================================================================
}  // namespace tatooine::reflection
//==============================================================================
#endif
