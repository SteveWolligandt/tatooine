#ifndef TATOOINE_ODE_SOLVER_H
#define TATOOINE_ODE_SOLVER_H
//==============================================================================
#include <cassert>
#include <concepts>
#include <tatooine/concepts.h>
#include <map>
#include <tatooine/field.h>
#include <tatooine/line.h>
//==============================================================================
namespace tatooine::ode {
//==============================================================================
template <typename F, typename Real, size_t N>
concept stepper_callback_invocable =
    std::regular_invocable<F, Real, vec<Real, N> const&, vec<Real, N> const&> ||
    std::regular_invocable<F, Real, vec<Real, N> const&>;
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
 private:
  //----------------------------------------------------------------------------
  template <typename V, std::floating_point VReal, real_number Y0Real,
            real_number T0Real, real_number TauReal,
            stepper_callback_invocable<Y0Real, N> StepperCallback>
  constexpr auto solve(vectorfield<V, VReal, N> const& v, vec<Y0Real, N>& y0,
                       T0Real t0, TauReal tau,
                       StepperCallback&& callback) const {
    as_derived().solve(v, y0, t0, tau, std::forward<StepperCallback>(callback));
  }
};
//==============================================================================
}  // namespace tatooine::ode
//==============================================================================
#endif
