#ifndef TATOOINE_RAY_H
#define TATOOINE_RAY_H

#include "tensor.h"
#include "type_traits.h"

//==============================================================================
namespace tatooine{
//==============================================================================

template <typename Real, size_t N>
struct ray {
  vec<Real, N> origin;
  vec<Real, N> direction;

  //============================================================================
  template <typename Real0, typename Real1>
  ray(const vec<Real0, N>& o, const vec<Real1, N>& d)
      : origin{o}, direction{d} {}

  //----------------------------------------------------------------------------
  template <typename Real0, typename Real1>
  ray(vec<Real0, N>&& o, vec<Real1, N>&& d)
      : origin{std::move(o)}, direction{std::move(d)} {}

  //============================================================================
  auto at(Real t) const { return origin + t * direction; }
  auto operator()(Real t) const { return at(t); }
};

//==============================================================================
// deduction guides
//==============================================================================
template <typename Real0, typename Real1, size_t N>
ray(const vec<Real0, N>&, const vec<Real1, N>&)
    ->ray<promote_t<Real0, Real1>, N>;

template <typename Real0, typename Real1, size_t N>
ray(vec<Real0, N>&&, vec<Real1, N>&&)
    ->ray<promote_t<Real0, Real1>, N>;

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
