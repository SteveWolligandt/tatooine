#ifndef TATOOINE_BOOST_RKCK54_INTEGRATOR_H
#define TATOOINE_BOOST_RKCK54_INTEGRATOR_H

#include <boost/numeric/odeint.hpp>
#include "../../tensor.h"
#include "boostintegrator.h"


//==============================================================================
namespace tatooine::integration::boost {
//==============================================================================

template <typename Real, size_t N>
struct rkck54_type {
  using stepper_t =
      ::boost::numeric::odeint::runge_kutta_cash_karp54<vec<Real, N>>;
  using t = ::boost::numeric::odeint::controlled_runge_kutta<stepper_t>;
};

//------------------------------------------------------------------------------
template <typename Real, size_t N>
struct rungekuttacashkarp54
    : integrator<Real, N, typename rkck54_type<Real, N>::t> {
  rungekuttacashkarp54(Real initial_stepsize         = 0.1,
                       Real absolute_error_tolerance = 1e-4,
                       Real relative_error_tolerance = 1e-4, Real a_x = 1,
                       Real a_dxdt = 1)
      : integrator<Real, N, typename rkck54_type<Real, N>::t>(
            typename rkck54_type<Real, N>::t(
                ::boost::numeric::odeint::default_error_checker<
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
