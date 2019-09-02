#ifndef __TATOOINE_BOOST_RKCK54_INTEGRATOR_H__
#define __TATOOINE_BOOST_RKCK54_INTEGRATOR_H__

#include <boost/numeric/odeint.hpp>
#include "../../vecmat.h"
#include "boostintegrator.h"

#include "../../concept_defines.h"

//==============================================================================
namespace tatooine::integration::boost {
//==============================================================================

template <typename real_t, std::size_t n>
struct rkck54_type {
  using stepper_t =
      ::boost::numeric::odeint::runge_kutta_cash_karp54<Vec<real_t, n>>;
  using t = ::boost::numeric::odeint::controlled_runge_kutta<stepper_t>;
};

//------------------------------------------------------------------------------
template <std::size_t n, typename real_t>
struct rungekuttacashkarp54
    : BoostIntegrator<n, real_t, typename rkck54_type<real_t, n>::t> {
  rungekuttacashkarp54(real_t initial_stepsize         = 0.01,
                       real_t absolute_error_tolerance = 1e-10,
                       real_t relative_error_tolerance = 1e-6, real_t a_x = 1,
                       real_t a_dxdt = 1)
      : BoostIntegrator<n, real_t, typename rkck54_type<real_t, n>::t>(
            typename rkck54_type<real_t, n>::t(
                ::boost::numeric::odeint::default_error_checker<
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
