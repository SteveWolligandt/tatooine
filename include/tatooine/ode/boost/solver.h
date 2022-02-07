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
  using this_t   = solver<Real, N, Stepper>;
  using parent_type = ode::solver<this_t, Real, N>;
  using typename parent_type::pos_t;
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
  solver(Stepper &&stepper, const Real stepsize)
      : m_stepper{std::move(stepper)}, m_stepsize{stepsize} {}
  //============================================================================
  template <arithmetic Y0Real, arithmetic T0Real, arithmetic TauReal,
            typename Evaluator,
            stepper_callback_invocable<Real, N> StepperCallback>
  constexpr void solve(Evaluator&& evaluator, vec<Y0Real, N> const& y0,
                       T0Real const t0, TauReal tau,
                       StepperCallback&& callback) const {
    constexpr auto callback_takes_derivative =
        std::is_invocable_v<StepperCallback, pos_t, Real, vec_t>;

    if (tau == 0) {
      return;
    }
    pos_t x_copy(y0);

    ::boost::numeric::odeint::integrate_adaptive(
        m_stepper,
        [&evaluator, tau, t0](pos_t const& y, pos_t& sample, Real t) {
          sample = evaluator(y, t);
        },
        x_copy, t0, t0 + tau, tau > 0 ? m_stepsize : -m_stepsize,
        [tau, t0, &callback, &evaluator](const pos_t& y, Real t) {
          if constexpr (!callback_takes_derivative) {
            callback(y, t);
          } else {
            callback(y, t, evaluator(y, t));
          }
        });
  }
  //----------------------------------------------------------------------------
  auto stepsize() -> auto& { return m_stepsize; }
  auto stepsize() const { return m_stepsize; }
};

//==============================================================================
}  // namespace tatooine::ode::boost
//==============================================================================

#endif
