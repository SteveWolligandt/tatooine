#ifndef __TATOOINE_RK4_INTEGRATOR_H__
#define __TATOOINE_RK4_INTEGRATOR_H__

#include "boostintegrator.h"

#include "../../concept_defines.h"

//==============================================================================
namespace tatooine::integration::boost {
//==============================================================================

template <typename real_t, size_t n>
struct rk4_helper {
  using type = ::boost::numeric::odeint::runge_kutta4<Vec<real_t, n>>;
};
template <typename real_t, size_t n>
using rk4_t = typename rk4_helper<real_t, n>::type;

//==============================================================================

template <size_t n, typename real_t>
struct rungekutta4
    : BoostIntegrator<n, real_t, rk4_t<real_t, n>> {
  rungekutta4(real_t stepsize = 0.01)
      : BoostIntegrator<n, real_t, rk4_t<real_t, n>>{rk4_t<real_t, n>{},
                                                     stepsize} {}
};

//==============================================================================
}  // namespace tatooine::integration::boost
//==============================================================================

#include "../../concept_undefs.h"
#endif
