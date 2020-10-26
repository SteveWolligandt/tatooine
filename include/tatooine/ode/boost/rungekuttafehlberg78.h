#ifndef TATOOINE_ODE_BOOST_RUNGEKUTTAFEHLBERG78_H
#define TATOOINE_ODE_BOOST_RUNGEKUTTAFEHLBERG78_H
//==============================================================================
#include <tatooine/ode/boost/solver.h>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>
//==============================================================================
namespace tatooine::ode::boost {
//==============================================================================
template <typename Real, size_t N>
struct rkf78_helper {
  using stepper_t =
      ::boost::numeric::odeint::runge_kutta_fehlberg78<vec<Real, N>>;
  using type = ::boost::numeric::odeint::controlled_runge_kutta<stepper_t>;
};
template <typename Real, size_t N>
using rkf78_t = typename rkf78_helper<Real, N>::type;

template <typename Real, size_t N>
struct rungekuttafehlberg78 : tatooine::ode::boost::solver<Real, N, rkf78_t<Real, N>> {
  rungekuttafehlberg78(const Real initial_stepsize         = 0.01,
                       const Real absolute_error_tolerance = 1e-6,
                       const Real relative_error_tolerance = 1e-6,
                       const Real a_x = 1, const Real a_dxdt = 1)
      : tatooine::ode::boost::solver<Real, N, rkf78_t<Real, N>>(
            rkf78_t<Real, N>(::boost::numeric::odeint::default_error_checker<
                             Real, ::boost::numeric::odeint::range_algebra,
                             ::boost::numeric::odeint::default_operations>(
                absolute_error_tolerance, relative_error_tolerance, a_x,
                a_dxdt)),
            initial_stepsize) {}
};
//==============================================================================
}  // namespace tatooine::ode::boost
//==============================================================================
#endif
