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
struct is_stepper_callback_invocable_impl : std::integral_constant<bool, 
    std::is_invocable_v<F, vec<Real, N> const&, Real, vec<Real, N> const&> ||
    std::is_invocable_v<F, vec<Real, N> const&, Real>>{};
template <typename F, typename Real, size_t N>
static constexpr is_stepper_callback_invocable = is_stepper_callback_invocable_impl<F, Real, N>::value;
template <typename F, typename Real, size_t N>
struct is_stepper_evaluator_impl : std::integral_constant<bool,
    !std::is_base_of_v<parent::field<Real, N, N>, std::decay_t<F>> &&
    std::is_invocable_v<F, vec<Real, N> const&, Real> &&
    (std::is_same_v<vec<Real, N>,
                    std::invoke_result_t<F, vec<Real, N>, Real>> ||
     std::is_same_v<std::optional<vec<Real, N>>,
                    std::invoke_result_t<F, vec<Real, N>, Real>>)>{};
template <typename F, typename Real, size_t N>
static constexpr is_stepper_evaluator = is_stepper_evaluator_impl<F, Real, N>::value;
#ifdef __cpp_concepts
template <typename F, typename Real, size_t N>
concept stepper_callback_invocable =
    std::regular_invocable<F, vec<Real, N> const&, Real, vec<Real, N> const&> ||
    std::regular_invocable<F, vec<Real, N> const&, Real>;
template <typename F, typename Real, size_t N>
concept stepper_evaluator =
    !std::is_base_of_v<parent::field<Real, N, N>, std::decay_t<F>> &&
    std::regular_invocable<F, vec<Real, N> const&, Real> &&
    (std::is_same_v<vec<Real, N>,
                    std::invoke_result_t<F, vec<Real, N>, Real>> ||
     std::is_same_v<std::optional<vec<Real, N>>,
                    std::invoke_result_t<F, vec<Real, N>, Real>>);
#endif
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
#ifdef __cpp_concepts
  template <typename V, std::floating_point VReal, arithmetic Y0Real,
            arithmetic T0Real, arithmetic TauReal,
            stepper_callback_invocable<Y0Real, N> StepperCallback>
#else
  template <typename V, typename VReal, typename Y0Real,
            typename T0Real, typename TauReal,
            stepper_callback_invocable<Y0Real, N> StepperCallback,
            enable_if<is_floating_point<VReal>> = true,
            enable_if<is_arithmetic<Y0Real, T0Real, TauReal>> = true,
            enable_if<is_stepper_callback_invocable<StepperCallback, Y0Real, N>> = true>
#endif
  constexpr auto solve(vectorfield<V, VReal, N> const& v, vec<Y0Real, N>& y0,
                       T0Real t0, TauReal tau,
                       StepperCallback&& callback) const {
    as_derived().solve(v, y0, t0, tau, std::forward<StepperCallback>(callback));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic Y0Real, arithmetic T0Real, arithmetic TauReal,
            stepper_evaluator<Y0Real, N>          Evaluator,
            stepper_callback_invocable<Y0Real, N> StepperCallback>
#else
  template <typename Y0Real, typename T0Real, typename TauReal,
            typename Evaluator, typename StepperCallback,
            enable_if<is_arithmetic<Y0Real, TauReal>> = true,
            enable_if<is_stepper_evaluator<Evaluator, Y0Real, N>> = true,
            enable_if<is_stepper_callback_invocable<StepperCallback, Y0Real, N>> = true>
#endif
  constexpr auto solve(Evaluator&& evaluator, vec<Y0Real, N>& y0, T0Real t0,
                       TauReal tau, StepperCallback&& callback) const {
    as_derived().solve(std::forward<Evaluator>(evaluator), y0, t0, tau,
                       std::forward<StepperCallback>(callback));
  }
};
//==============================================================================
}  // namespace tatooine::ode
//==============================================================================
#endif
