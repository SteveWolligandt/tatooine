#ifndef TATOOINE_ODE_BOOST_RUNGEKUTTA4_H
#define TATOOINE_ODE_BOOST_RUNGEKUTTA4_H
//==============================================================================
#include <tatooine/ode/boost/solver.h>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
//==============================================================================
namespace tatooine::ode::boost {
//==============================================================================
template <typename Real, size_t N>
struct rk4_helper {
  using type = ::boost::numeric::odeint::runge_kutta4<vec<Real, N>>;
};
template <typename Real, size_t N>
using rk4_t = typename rk4_helper<Real, N>::type;
//==============================================================================
template <typename Real, size_t N>
struct rungekutta4 : tatooine::ode::boost::solver<Real, N, rk4_t<Real, N>> {
  rungekutta4(Real stepsize = 0.1)
      : tatooine::ode::boost::solver<Real, N, rk4_t<Real, N>>(
            rk4_t<Real, N>{}, stepsize) {}
};
//==============================================================================
}  // namespace tatooine::integration::boost
//==============================================================================
#endif
