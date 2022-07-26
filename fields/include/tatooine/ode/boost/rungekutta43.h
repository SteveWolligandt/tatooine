#ifndef TATOOINE_ODE_BOOST_RUNGEKUTTA43_H
#define TATOOINE_ODE_BOOST_RUNGEKUTTA43_H
//==============================================================================
#include <tatooine/ode/boost/solver.h>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
//==============================================================================
namespace tatooine::ode::boost {
//==============================================================================
template <typename Real, size_t N>
struct rk43_helper {
  using stepper = ::boost::numeric::odeint::runge_kutta4<vec<Real, N>>;
  using type    = ::boost::numeric::odeint::controlled_runge_kutta<stepper>;
};
template <typename Real, size_t N>
using rk43_t = typename rk43_helper<Real, N>::type;

template <typename Real, size_t N>
struct rungekutta43 : tatooine::ode::boost::solver<Real, N, rk43_t<Real, N>> {
  rungekutta43(const Real absolute_error_tolerance = 1e-6,
                       const Real relative_error_tolerance = 1e-6,
                       const Real initial_stepsize         = 0.01,
                       const Real a_x = 1, const Real a_dxdt = 1)
      : tatooine::ode::boost::solver<Real, N, rk43_t<Real, N>>(
            rk43_t<Real, N>(::boost::numeric::odeint::default_error_checker<
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
