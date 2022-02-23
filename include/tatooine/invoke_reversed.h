#ifndef TATOOINE_INVOKE_REVERSED_H
#define TATOOINE_INVOKE_REVERSED_H
//==============================================================================
#include <functional>
#include <utility>
//==============================================================================
namespace tatooine {
//==============================================================================
auto constexpr invoke_reversed(auto&& f, auto&& param0)
    -> decltype(auto) requires std::invocable<decltype(f), decltype(param0)> {
  return std::invoke(f, std::forward<decltype(param0)>(param0));
}
//------------------------------------------------------------------------------
auto constexpr invoke_reversed(auto&& f, auto&& param0, auto&& param1,
                               auto&&... params) -> decltype(auto) {
  return invoke_reversed(
      [&](auto&&... params) -> decltype(auto) {
        return std::invoke(std::forward<decltype(f)>(f),
                           std::forward<decltype(params)>(params)...,
                           std::forward<decltype(param0)>(param0));
      },
      std::forward<decltype(param1)>(param1),
      std::forward<decltype(params)>(params)...);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
