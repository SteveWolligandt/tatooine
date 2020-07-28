#ifndef TATOOINE_VARIADIC_HELPERS_H
#define TATOOINE_VARIADIC_HELPERS_H

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename... Ts>
struct front_t_impl {
  using type = T;
};
template <typename... Ts>
using front_t = typename front_t_impl<Ts...>::type;

//==============================================================================
template <typename... T>
struct back_t_impl;
template <typename T>
struct back_t_impl<T> {
  using type = T;
};
template <typename T, typename... Ts>
struct back_t_impl<T, Ts...> {
  using type = typename back_t_impl<Ts...>::type;
};
template <typename... Ts>
using back_t = typename back_t_impl<Ts...>::type;

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
