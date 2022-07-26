#ifndef TATOOINE_ODE_BOOST_RUNGEKUTTADOPRI5_H
#define TATOOINE_ODE_BOOST_RUNGEKUTTADOPRI5_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/ode/boost/domain_error_checker.h>
#include <tatooine/ode/boost/solver.h>

#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
//==============================================================================
namespace tatooine::ode::boost {
//==============================================================================
template <typename Real, std::size_t N>
struct rkd5_aux {
  using stepper_type =
      ::boost::numeric::odeint::runge_kutta_dopri5<vec<Real, N>>;
  using error_checker_type =
      domain_error_checker<Real, typename stepper_type::algebra_type,
                           typename stepper_type::operations_type>;
  using controller_type =
      ::boost::numeric::odeint::controlled_runge_kutta<stepper_type,
                                                       error_checker_type>;
};
//==============================================================================
template <typename Real, std::size_t N>
struct rungekuttadopri5
    : tatooine::ode::boost::solver<
          Real, N, typename rkd5_aux<Real, N>::controller_type> {
  using controller_type    = typename rkd5_aux<Real, N>::controller_type;
  using error_checker_type = typename rkd5_aux<Real, N>::error_checker_type;
  rungekuttadopri5(Real const absolute_error_tolerance = 1e-10,
                   Real const relative_error_tolerance = 1e-6,
                   Real const initial_stepsize = 0.01, Real const a_x = 1,
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
