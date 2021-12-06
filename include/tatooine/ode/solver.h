#ifndef TATOOINE_ODE_SOLVER_H
#define TATOOINE_ODE_SOLVER_H
//==============================================================================
#include <cassert>
#include <tatooine/concepts.h>
#include <map>
#include <tatooine/field.h>
#include <tatooine/line.h>
//==============================================================================
namespace tatooine::ode {
//==============================================================================
template <typename F, typename Real, size_t N>
struct is_stepper_callback_invocable_impl
    : std::integral_constant<
          bool, std::is_invocable_v<F, vec<Real, N> const&, Real,
                                    vec<Real, N> const&> ||
                    std::is_invocable_v<F, vec<Real, N> const&, Real>> {};
template <typename F, typename Real, size_t N>
struct is_stepper_evaluator_impl
    : std::integral_constant<
          bool,
          !std::is_base_of_v<polymorphic::vectorfield<Real, N>, std::decay_t<F>> &&
              std::is_invocable_v<F, vec<Real, N> const&, Real> &&
              (std::is_same_v<vec<Real, N>,
                              std::invoke_result_t<F, vec<Real, N>, Real>> ||
               std::is_same_v<std::optional<vec<Real, N>>,
                              std::invoke_result_t<F, vec<Real, N>, Real>>)> {};

template <typename F, typename Real, size_t N>
static constexpr auto is_stepper_callback_invocable =
    is_stepper_callback_invocable_impl<F, Real, N>::value;

template <typename F, typename Real, size_t N>
static constexpr auto is_stepper_evaluator =
    is_stepper_evaluator_impl<F, Real, N>::value;
template <typename F, typename Real, size_t N>
concept stepper_callback_invocable =
    std::regular_invocable<F, vec<Real, N> const&, Real, vec<Real, N> const&> ||
    std::regular_invocable<F, vec<Real, N> const&, Real>;
template <typename F, typename Real, size_t N>
concept stepper_evaluator =
    !std::is_base_of_v<polymorphic::vectorfield<Real, N>, std::decay_t<F>> &&
    std::regular_invocable<F, vec<Real, N> const&, Real> &&
    (std::is_same_v<vec<Real, N>,
                    std::invoke_result_t<F, vec<Real, N>, Real>> ||
     std::is_same_v<std::optional<vec<Real, N>>,
                    std::invoke_result_t<F, vec<Real, N>, Real>>);
//==============================================================================
template <typename Derived, typename Real, size_t N>
struct solver : crtp<Derived> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using parent_t = crtp<Derived>;
  using parent_t::as_derived;
  using vec_t = vec<Real, N>;
  using pos_t = vec_t;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  template <typename V, std::floating_point VReal, arithmetic Y0Real,
            arithmetic T0Real, arithmetic TauReal,
            stepper_callback_invocable<Y0Real, N> StepperCallback>
  constexpr auto solve(vectorfield<V, VReal, N> const& v, vec<Y0Real, N>& y0,
                       T0Real t0, TauReal tau,
                       StepperCallback&& callback) const {
    as_derived().solve(v, y0, t0, tau, std::forward<StepperCallback>(callback));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <arithmetic Y0Real, arithmetic T0Real, arithmetic TauReal,
            stepper_evaluator<Y0Real, N>          Evaluator,
            stepper_callback_invocable<Y0Real, N> StepperCallback>
  constexpr auto solve(Evaluator&& evaluator, vec<Y0Real, N>& y0, T0Real t0,
                       TauReal tau, StepperCallback&& callback) const {
    as_derived().solve(std::forward<Evaluator>(evaluator), y0, t0, tau,
                       std::forward<StepperCallback>(callback));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t K, arithmetic Y0Real, arithmetic T0Real, arithmetic TauReal,
            stepper_evaluator<Y0Real, N>          Evaluator,
            stepper_callback_invocable<Y0Real, N> StepperCallback>
  constexpr auto solve(Evaluator&& evaluator, mat<Y0Real, N, K>& y0s, T0Real t0,
                       TauReal tau, StepperCallback&& callback) const {
    as_derived().solve(std::forward<Evaluator>(evaluator), y0s, t0, tau,
                       std::forward<StepperCallback>(callback));
  }
};
//==============================================================================
}  // namespace tatooine::ode
//==============================================================================
#endif
