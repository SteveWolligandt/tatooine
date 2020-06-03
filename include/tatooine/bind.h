#ifndef TATOOINE_BIND_H
#define TATOOINE_BIND_H

#include <functional>
#include "cxxstd.h"

//==============================================================================
namespace tatooine {
//==============================================================================
/// binds first arguments of f (either all or only partially)
template <typename F, typename... Args>
constexpr auto bind(F&& f, Args&&... args) {
  return [&](auto&&... rest) -> decltype(auto) {
    return f(std::forward<Args>(args)...,
             std::forward<decltype(rest)>(rest)...);
  };
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
