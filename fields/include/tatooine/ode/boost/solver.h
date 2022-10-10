#ifndef TATOOINE_ODE_BOOST_SOLVER_H
#define TATOOINE_ODE_BOOST_SOLVER_H
//==============================================================================
#include <tatooine/ode/solver.h>
#include <tatooine/tensor.h>

#include <boost/numeric/odeint.hpp>
//==============================================================================
namespace boost::numeric::odeint {
//==============================================================================
template <typename Real, size_t N>
struct is_resizeable<tatooine::vec<Real, N>> {
  using type              = boost::false_type;
  static const bool value = type::value;
};
//==============================================================================
}  // namespace boost::numeric::odeint
//==============================================================================
namespace tatooine::ode::boost {
//==============================================================================
template <typename Real, size_t N, typename Stepper>
struct solver : ode::solver<solver<Real, N, Stepper>, Real, N> {
 public:
  using this_type   = solver<Real, N, Stepper>;
  using parent_type = ode::solver<this_type, Real, N>;
  using typename parent_type::pos_type;
  using typename parent_type::vec_t;

 protected:
  //============================================================================
  Stepper m_stepper;
  Real    m_stepsize;

 private:
  //============================================================================
  friend parent_type;

 public:
  //============================================================================
  solver(const Stepper &stepper, const Real stepsize)
      : m_stepper{stepper}, m_stepsize{stepsize} {}
  solver(Stepper&& stepper, const Real stepsize)
      : m_stepper{std::move(stepper)}, m_stepsize{stepsize} {}
  //============================================================================
  template <arithmetic                          Y0Real, typename Evaluator,
            stepper_callback_invocable<Real, N> StepperCallback>
  constexpr void solve(Evaluator&& evaluator, vec<Y0Real, N> const& y0,
                       arithmetic auto const t0, arithmetic auto tau,
                       StepperCallback&& callback) const {
    constexpr auto callback_takes_derivative =
        std::is_invocable_v<StepperCallback, pos_type, Real, vec_t>;

    if (tau == 0) {
      return;
    }
    auto x_copy = pos_type{y0};
    try {
    ::boost::numeric::odeint::integrate_adaptive(
        m_stepper,
        [&evaluator, tau, t0](pos_type const& y, pos_type& sample, Real t) {
          sample = evaluator(y, t);
        },
        x_copy, Real(t0), Real(t0 + tau), Real(tau > 0 ? m_stepsize : -m_stepsize),
        [tau, t0, &callback, &evaluator](const pos_type& y, Real t) {
          if constexpr (!callback_takes_derivative) {
            callback(y, t);
          } else {
            callback(y, t, evaluator(y, t));
          }
        });
    } catch (::boost::numeric::odeint::step_adjustment_error const&) {
      if constexpr (!callback_takes_derivative) {
        callback(pos_type::fill(nan()), nan());
      } else {
        using derivative_type = decltype(evaluator(y0, t0));
        callback(pos_type::fill(nan()), nan(), derivative_type::fill(nan()));
      }
    }
  }
  //----------------------------------------------------------------------------
  auto stepsize() -> auto& { return m_stepsize; }
  auto stepsize() const { return m_stepsize; }
};

//==============================================================================
}  // namespace tatooine::ode::boost
//==============================================================================

#endif
