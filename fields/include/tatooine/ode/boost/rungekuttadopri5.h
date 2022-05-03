#ifndef __TATOOINE_RKD5_INTEGRATOR_H__
#define __TATOOINE_RKD5_INTEGRATOR_H__

#include "boostintegrator.h"


//==============================================================================
namespace tatooine::integration::boost {
//==============================================================================

template <typename Real, size_t N>
struct rkd5_helper {
  using stepper_t =
      ::boost::numeric::odeint::runge_kutta_dopri5<vec<Real, N>>;
  using type = ::boost::numeric::odeint::controlled_runge_kutta<stepper_t>;
};
template <typename Real, size_t N>
using rkd5_t = typename rkd5_helper<Real, N>::type;
template <typename Real, size_t N>
using rkd5_stepper = typename rkd5_helper<Real, N>::stepper_t;

//==============================================================================

template <typename Real, size_t N>
struct rungekuttadopri5 : integrator<Real, N, rkd5_t<Real, N>> {
  rungekuttadopri5(Real initial_stepsize         = 0.01,
                   Real absolute_error_tolerance = 1e-10,
                   Real relative_error_tolerance = 1e-6, Real a_x = 1,
                   Real a_dxdt = 1)
      : integrator<Real, N, rkd5_t<Real, N>>(
            rkd5_t<Real, N>(::boost::numeric::odeint::default_error_checker<
                            Real, ::boost::numeric::odeint::range_algebra,
                            ::boost::numeric::odeint::default_operations>(
                absolute_error_tolerance, relative_error_tolerance, a_x,
                a_dxdt)),
            initial_stepsize) {}
};

//==============================================================================
}  // namespace tatooine::integration::boost
//==============================================================================

#endif
