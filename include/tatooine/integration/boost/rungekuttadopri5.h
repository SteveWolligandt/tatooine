#ifndef __TATOOINE_RKD5_INTEGRATOR_H__
#define __TATOOINE_RKD5_INTEGRATOR_H__

#include "boostintegrator.h"

#include "../../concept_defines.h"

//==============================================================================
namespace tatooine::integration::boost {
//==============================================================================

template <typename real_t, size_t n>
struct rkd5_helper {
  using stepper_t =
      ::boost::numeric::odeint::runge_kutta_dopri5<Vec<real_t, n>>;
  using type = ::boost::numeric::odeint::controlled_runge_kutta<stepper_t>;
};
template <typename real_t, size_t n>
using rkd5_t = typename rkd5_helper<real_t, n>::type;
template <typename real_t, size_t n>
using rkd5_stepper = typename rkd5_helper<real_t, n>::stepper_t;

//==============================================================================

template <size_t n, typename real_t, interpolator_concept interpolator_t>
struct RungeKuttaDopri5
    : BoostIntegrator<n, real_t, interpolator_t, rkd5_t<real_t, n>> {
  RungeKuttaDopri5(real_t initial_stepsize         = 0.01,
                   real_t absolute_error_tolerance = 1e-10,
                   real_t relative_error_tolerance = 1e-6, real_t a_x = 1,
                   real_t a_dxdt = 1)
      : BoostIntegrator<n, real_t, interpolator_t, rkd5_t<real_t, n>>(
            rkd5_t<real_t, n>(::boost::numeric::odeint::default_error_checker<
                              real_t, ::boost::numeric::odeint::range_algebra,
                              ::boost::numeric::odeint::default_operations>(
                absolute_error_tolerance, relative_error_tolerance, a_x,
                a_dxdt)),
            initial_stepsize) {}
};

//==============================================================================
}  // namespace tatooine::integration::boost
//==============================================================================

#include "../../concept_undefs.h"
#endif
