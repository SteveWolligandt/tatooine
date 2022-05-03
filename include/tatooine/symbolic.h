#ifndef TATOOINE_SYMBOLIC_H
#define TATOOINE_SYMBOLIC_H
#include <tatooine/available_libraries.h>
#if TATOOINE_GINAC_AVAILABLE
//==============================================================================
#include <ginac/ginac.h>
//==============================================================================
namespace tatooine::symbolic {
//==============================================================================
#define sym(sym)                                                           \
  template <size_t... Is>                                                  \
  static auto sym(size_t idx, std::index_sequence<Is...> /*is*/)->auto& {  \
    static std::array<GiNaC::symbol, sizeof...(Is)> sym_arr{               \
        ((void)Is,                                                         \
         GiNaC::symbol{#sym + std::string{"_"} + std::to_string(Is)})...}; \
    return sym_arr[idx];                                                   \
  }                                                                        \
  static auto sym(size_t idx)->auto& {                                     \
    return sym(idx, std::make_index_sequence<num_pos_symbols>{});          \
  }                                                                        \
  static auto sym()->auto& {                                               \
    static GiNaC::symbol sym{#sym};                                        \
    return sym;                                                            \
  }

struct symbol {
  static constexpr size_t num_pos_symbols = 100;
  sym(i) sym(j) sym(k) sym(x) sym(y) sym(z) sym(t)
};
#undef sym
#undef symarr
//----------------------------------------------------------------------------
/// substitudes expression with relations
template <typename... Relations>
auto ev(const GiNaC::ex& expr, Relations&&... relations) {
  GiNaC::lst substitutions{std::forward<Relations>(relations)...};
  return expr.subs(substitutions);
}
//----------------------------------------------------------------------------
/// substitudes expression with relations and casts substituted expression to
/// double
template <typename out_real_type = double, typename... Relations>
auto evtod(const GiNaC::ex& expr, Relations&&... relations) {
  return static_cast<out_real_type>(
      GiNaC::ex_to<GiNaC::numeric>(
          ev(expr, std::forward<Relations>(relations)...).evalf())
          .to_double());
}
//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================
// type_traits
//==============================================================================
#include "type_traits.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
struct is_symbolic
    : std::integral_constant<bool, std::is_same_v<T, GiNaC::ex> ||
                                       std::is_same_v<T, GiNaC::symbol>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_symbolic_v = is_symbolic<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
struct are_symbolic;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_symbolic_v = are_symbolic<Ts...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_symbolic<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_symbolic<T> : std::integral_constant<bool, is_symbolic_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_symbolic<T0, T1, Ts...>
    : std::integral_constant<bool, are_symbolic_v<T0> &&
                                       are_symbolic_v<T1, Ts...>> {};
//------------------------------------------------------------------------------
template <typename... Ts>
struct are_arithmetic_or_symbolic
    : std::integral_constant<bool, are_arithmetic_v<Ts...> ||
                                   are_symbolic_v<Ts...>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_arithmetic_or_symbolic_v =
    are_arithmetic_or_symbolic<Ts...>::value;
//==============================================================================
template <typename... Ts>
struct are_arithmetic_complex_or_symbolic
    : std::integral_constant<bool, are_arithmetic_v<Ts...> ||
                                       are_complex_v<Ts...> ||
                                       are_symbolic_v<Ts...>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_arithmetic_complex_or_symbolic_v =
    are_arithmetic_complex_or_symbolic<Ts...>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
#endif
