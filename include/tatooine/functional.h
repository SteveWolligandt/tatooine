#ifndef TATOOINE_FUNCTIONAL_H
#define TATOOINE_FUNCTIONAL_H

#include <functional>

#include <tatooine/invoke_omitted.h>
#include <tatooine/invoke_unpacked.h>
#include <tatooine/map.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename... Ts>
auto front_param(T&& head, Ts&&... /*tail*/) -> decltype(auto) {
  return std::forward<T>(head);
}
//==============================================================================
template <typename T>
auto back_param(T&& t) -> decltype(auto) {
  return std::forward<T>(t);
}
//==============================================================================
template <typename T0, typename T1, typename... Ts>
auto back_param(T0&& /*t0*/, T1&& t1, Ts&&... ts) -> decltype(auto) {
  return back_param(std::forward<T1>(t1), std::forward<Ts>(ts)...);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
