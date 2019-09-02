#ifndef __TATOOINE_RKF78_INTEGRATOR_H__
#define __TATOOINE_RKF78_INTEGRATOR_H__

#include "boostintegrator.h"

namespace tatooine::integration::boost {
template <typename real_t, size_t n>
struct rkf78_helper {
  using stepper_t =
      ::boost::numeric::odeint::runge_kutta_fehlberg78<Vec<real_t, n>>;
  using type = ::boost::numeric::odeint::controlled_runge_kutta<stepper_t>;
};
template <typename real_t, size_t n>
using rkf78_t = typename rkf78_helper<real_t, n>::type;

template <size_t n, typename real_t, interpolator_concept interpolator_t>
struct RungeKuttaFehlberg78
    : BoostIntegrator<n, real_t, interpolator_t, rkf78_t<real_t>> {
  RungeKuttaFehlberg78(const real_t initial_stepsize         = 0.01,
                       const real_t absolute_error_tolerance = 1e-10,
                       const real_t relative_error_tolerance = 1e-6,
                       const real_t a_x = 1, const real_t a_dxdt = 1)
      : BoostIntegrator<n, real_t, interpolator_t, rkf78_t<real_t>>(
            rkf78_t<real_t>(                                      //
                ::boost::numeric::odeint::default_error_checker<  //
                    real_t,                                       //
                    ::boost::numeric::odeint::range_algebra,      //
                    ::boost::numeric::odeint::default_operations  //
                    >(absolute_error_tolerance, relative_error_tolerance, a_x,
                      a_dxdt)),
            initial_stepsize) {}
};
}  // namespace tatooine::integration::boost

#endif
