#ifndef TATOOINE_ODE_BOOST_RUNGEKUTTAFEHLBERG78_H
#define TATOOINE_ODE_BOOST_RUNGEKUTTAFEHLBERG78_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/ode/boost/domain_error_checker.h>
#include <tatooine/ode/boost/solver.h>

#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>
//==============================================================================
namespace tatooine::ode::boost {
//==============================================================================
template <floating_point Real, std::size_t N>
struct rkf78_aux {
  using stepper_type =
      ::boost::numeric::odeint::runge_kutta_fehlberg78<vec<Real, N>>;
  using error_checker_type =
      domain_error_checker<Real, typename stepper_type::algebra_type,
                           typename stepper_type::operations_type>;
  using controller_type =
      ::boost::numeric::odeint::controlled_runge_kutta<stepper_type,
                                                       error_checker_type>;
};
//==============================================================================
template <floating_point Real, std::size_t N>
struct rungekuttafehlberg78
    : tatooine::ode::boost::solver<
          Real, N, typename rkf78_aux<Real, N>::controller_type> {
  using controller_type    = typename rkf78_aux<Real, N>::controller_type;
  using error_checker_type =  typename rkf78_aux<Real, N>::error_checker_type;
  rungekuttafehlberg78(Real const absolute_error_tolerance = 1e-10,
                       Real const relative_error_tolerance = 1e-6,
                       Real const initial_stepsize = 1e-2, Real const a_x = 1,
                       Real const a_dxdt = 1)
      : tatooine::ode::boost::solver<Real, N, controller_type>(
            controller_type{error_checker_type{absolute_error_tolerance,
                                               relative_error_tolerance, a_x,
                                               a_dxdt}},
            initial_stepsize) {}
};
//==============================================================================
}  // namespace tatooine::ode::boost
//==============================================================================
#endif
