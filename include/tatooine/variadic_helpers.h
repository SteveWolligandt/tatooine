#ifndef TATOOINE_VARIADIC_HELPERS_H
#define TATOOINE_VARIADIC_HELPERS_H

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename... Ts>
struct front {
  using type = T;
};
template <typename... Ts>
using front_t = typename front<Ts...>::type;

//==============================================================================
template <typename... T>
struct back;
template <typename T>
struct back<T> {
  using type = T;
};
template <typename T, typename... Ts>
struct back<T, Ts...> {
  using type = typename back<Ts...>::type;
};
template <typename... Ts>
using back_t = typename back<Ts...>::type;

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
