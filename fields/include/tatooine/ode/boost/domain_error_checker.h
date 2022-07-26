#ifndef TATOOINE_ODE_BOOST_DOMAIN_ERROR_CHECKER_H
#define TATOOINE_ODE_BOOST_DOMAIN_ERROR_CHECKER_H
//==============================================================================
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
//==============================================================================
namespace tatooine::ode::boost {
//==============================================================================
template <typename Value, typename Algebra, typename Operations>
class domain_error_checker
    : ::boost::numeric::odeint::default_error_checker<Value, Algebra,
                                                      Operations> {
 public:
  using parent_type =
      ::boost::numeric::odeint::default_error_checker<Value, Algebra,
                                                      Operations>;
  using value_type   = typename parent_type::value_type;
  using algebra_type = typename parent_type::algebra_type;
  //============================================================================
  domain_error_checker(value_type eps_abs = static_cast<value_type>(1.0e-10),
                       value_type eps_rel = static_cast<value_type>(1.0e-6),
                       value_type a_x     = static_cast<value_type>(1),
                       value_type a_dxdt  = static_cast<value_type>(1))
      : parent_type{eps_abs, eps_rel, a_x, a_dxdt} {}
  //============================================================================
  auto error(algebra_type &algebra, auto const &x_old, auto const &dxdt_old,
             auto &x_err, auto dt) const -> value_type {
    if (std::isnan(x_err(0))) {
      return 2;
    }
    return parent_type::error(algebra, x_old, dxdt_old, x_err, dt);
  }
};
//==============================================================================
}  // namespace tatooine::ode::boost
//==============================================================================
#endif
