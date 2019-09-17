#ifndef TATOOINE_BOOST_RK4_INTEGRATOR_H
#define TATOOINE_BOOST_RK4_INTEGRATOR_H

#include "boostintegrator.h"

//==============================================================================
namespace tatooine::integration::boost {
//==============================================================================

template <typename Real, size_t N>
struct rk4_helper {
  using type = ::boost::numeric::odeint::runge_kutta4<vec<Real, N>>;
};
template <typename Real, size_t N>
using rk4_t = typename rk4_helper<Real, N>::type;

//==============================================================================

template <typename Real, size_t N>
struct rungekutta4 : integrator<Real, N, rk4_t<Real, N>> {
  rungekutta4(Real stepsize = 0.1)
      : integrator<Real, N, rk4_t<Real, N>>{rk4_t<Real, N>{}, stepsize} {}
};

//==============================================================================
}  // namespace tatooine::integration::boost
//==============================================================================

#endif
