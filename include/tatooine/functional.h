#ifndef TATOOINE_FUNCTIONAL_H
#define TATOOINE_FUNCTIONAL_H

#include <functional>

#include "bind.h"
#include "cxxstd.h"
#include "invoke_omitted.h"
#include "invoke_unpacked.h"
#include "map.h"

//==============================================================================
template <typename T, typename... Ts>
decltype(auto) front_param(T&& head, Ts&&... /*tail*/) {
  return std::forward<T>(head);
}
//==============================================================================
template <typename T>
decltype(auto) back_param(T&& t) {
  return std::forward<T>(t);
}
//==============================================================================
template <typename T0, typename T1, typename... Ts>
decltype(auto) back_param(T0&& /*t0*/, T1&& t1, Ts&&... ts) {
  return back_param(std::forward<T1>(t1), std::forward<Ts>(ts)...);
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
