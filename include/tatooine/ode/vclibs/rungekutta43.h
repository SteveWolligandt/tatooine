#ifndef TATOOINE_ODE_VCLIBS_RK43_H
#define TATOOINE_ODE_VCLIBS_RK43_H
//==============================================================================
#include <tatooine/interpolation.h>
#include <tatooine/concepts.h>
#include <boost/range/numeric.hpp>
#include <vcode/odeint.hh>
#include <tatooine/ode/solver.h>
//==============================================================================
template <typename Real, size_t N>
struct VC::odeint::vector_operations_t<tatooine::vec<Real, N>> {
  using vec_t = tatooine::vec<Real, N>;

  //----------------------------------------------------------------------------
  static constexpr bool isfinitenorm(vec_t const& v) {
    for (auto c : v) {
      if (!std::isfinite(c)) { return false; }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  static constexpr Real norm2(vec_t const& x) {
    static_assert(std::is_floating_point<Real>(), "require floating point");
    return std::sqrt(sqr(x));
  }

  //----------------------------------------------------------------------------
  static constexpr Real norm1(vec_t const& x) { return tatooine::norm1(x); }
  //----------------------------------------------------------------------------
  static constexpr Real norminf(vec_t const& x) { return norm_inf(x); }
  //----------------------------------------------------------------------------
  static constexpr auto abs(vec_t v) {
    for (size_t i = 0; i < N; ++i) { v(i) = std::abs(v(i)); }
    return v;
  }

  //----------------------------------------------------------------------------
  static constexpr auto max(vec_t const& x, vec_t const& y) {
    vec_t v;
    for (size_t i = 0; i < N; ++i) { v(i) = std::max(x(i), y(i)); }
    return v;
  }
};

//==============================================================================
namespace tatooine::ode::vclibs {
//==============================================================================

static constexpr inline auto rk43          = VC::odeint::RK43;
static constexpr inline auto out_of_domain = VC::odeint::OutOfDomain;
static constexpr inline auto abs_tol       = VC::odeint::AbsTol;
static constexpr inline auto rel_tol       = VC::odeint::RelTol;
static constexpr inline auto initial_step  = VC::odeint::InitialStep;
static constexpr inline auto max_step      = VC::odeint::MaxStep;
static constexpr inline auto max_num_steps = VC::odeint::MaxNumSteps;

template <typename Real, size_t N>
struct rungekutta43 : solver<rungekutta43<Real, N>, Real, N> {
  //============================================================================
  using this_t   = rungekutta43<Real, N>;
  using parent_t = solver<this_t, Real, N>;
  using typename parent_t::pos_t;
  using vc_ode_t      = VC::odeint::ode_t<N, Real, vec<Real, N>, false>;
  using vc_stepper_t  = typename vc_ode_t::template solver_t<
      VC::odeint::steppers::rk43_t<pos_t, Real>>;
  using vc_options_t = typename vc_ode_t::options_t;

  //============================================================================
 private:
  mutable vc_stepper_t m_stepper;

  //============================================================================
 public:
  constexpr rungekutta43()
      : m_stepper{vc_ode_t::solver(rk43, vc_options_t{abs_tol = 1e-4, rel_tol = 1e-4,
                                   initial_step = 0, max_step = 0.1})} {}
  constexpr rungekutta43(rungekutta43 const& other)     = default;
  constexpr rungekutta43(rungekutta43&& other) noexcept = default;
  constexpr auto operator=(rungekutta43 const& other)
      -> rungekutta43&   = default;
  constexpr auto operator=(rungekutta43&& other) noexcept
      -> rungekutta43&   = default;
  //----------------------------------------------------------------------------
  template <typename... Options>
  rungekutta43(Options&&... options)
      : m_stepper{vc_ode_t::solver(
            rk43, vc_options_t{std::forward<Options>(options)...})} {}

  //============================================================================
  /// Continues integration of integral.
  /// if tau > 0 than it integrates forward and pushes new samples back
  /// otherwise pushes samples to front.
  template <typename V, std::floating_point VReal, real_number Y0Real,
            real_number T0Real, real_number TauReal,
            stepper_callback_invocable<Real, N> StepperCallback>
  constexpr void solve(vectorfield<V, VReal, N> const& v, vec<Y0Real, N> const& y0,
             T0Real const t0, TauReal tau, StepperCallback&& callback) const {
    if (tau == 0) { return; }
    // do not start integration if y0, t0 is not in domain of vectorfield
    if (!v.in_domain(y0, t0)) { return; }

    auto dy = [&v](Real t, pos_t const& y) -> typename vc_ode_t::maybe_vec {
      if (!v.in_domain(y, t)) {
        return out_of_domain;
      }
      return v(y, t);
    };

    m_stepper.initialize(dy, t0, t0 + tau, y0);

    m_stepper.integrate(
        dy, vc_ode_t::Output >>
                vc_ode_t::sink(std::forward<StepperCallback>(callback)));
  }
};
//==============================================================================
}  // namespace tatooine::ode::vclibs
//==============================================================================

#endif
