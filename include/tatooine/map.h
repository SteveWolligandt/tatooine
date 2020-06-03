#ifndef TATOOINE_MAP_H
#define TATOOINE_MAP_H
//==============================================================================
#include <functional>
#include "cxxstd.h"
//==============================================================================
namespace tatooine {
//==============================================================================
/// maps unary function f to all single parameters of parameter pack ts
template <typename... Ts, typename F>
constexpr void map(F&& f, Ts&&... ts) {
  (f(std::forward<Ts>(ts)), ...);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
