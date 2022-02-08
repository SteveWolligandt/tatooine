#ifndef TATOOINE_NEWTON_RAPHSON_H
#define TATOOINE_NEWTON_RAPHSON_H

#include "symbolic_field.h"
#include "diff.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N, size_t... Is>
[[nodiscard]] auto newton_raphson(const symbolic::field<Real, N, N>&          v,
                                  typename symbolic::field<Real, N, N>::pos_type x,
                                  Real t, size_t n, double precision,
                                  std::index_sequence<Is...>) {
  auto Jv     = diff(v);
  auto step = vec<GiNaC::ex, N>{v.x(Is)...} - inverse(Jv.expr()) * v.expr();

  for (size_t i = 0; i < n; ++i) {
    auto y = evtod<Real>(step, (v.x(Is) == x(Is))..., v.t() == t);
    if (distance(x, y) < precision) {
      x = std::move(y);
      break;
    };
    x = std::move(y);
  }
  return x;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
[[nodiscard]] auto newton_raphson(const symbolic::field<Real, N, N>&          v,
                                  typename symbolic::field<Real, N, N>::pos_type x,
                                  Real t, size_t n, double precision = 1e-10) {
  return newton_raphson(v, x, t, n, precision, std::make_index_sequence<N>{});
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
