#ifndef TATOOINE_ODE_VCLIBS_RK43_H
#define TATOOINE_ODE_VCLIBS_RK43_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/interpolation.h>
#include <tatooine/ode/solver.h>
#include <tatooine/type_traits.h>

#include <boost/range/numeric.hpp>
#include <vcode/odeint.hh>
//==============================================================================
template <typename Real, size_t N>
struct VC::odeint::vector_operations_t<tatooine::vec<Real, N>> {
  using vec_t = tatooine::vec<Real, N>;

  //----------------------------------------------------------------------------
  static constexpr bool isfinitenorm(vec_t const& v) {
    for (auto c : v) {
      if (!std::isfinite(c)) {
        return false;
      }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  static constexpr Real norm2(vec_t const& x) {
    static_assert(std::is_floating_point<Real>(), "require floating point");
    return std::sqrt(sqr(x));
  }

  //----------------------------------------------------------------------------
  static constexpr Real norm1(vec_t const& x) {
    return tatooine::norm1(x);
  }
  //----------------------------------------------------------------------------
  static constexpr Real norminf(vec_t const& x) {
    return norm_inf(x);
  }
  //----------------------------------------------------------------------------
  static constexpr auto abs(vec_t v) {
    for (size_t i = 0; i < N; ++i) {
      v(i) = std::abs(v(i));
    }
    return v;
  }

  //----------------------------------------------------------------------------
  static constexpr auto max(vec_t const& x, vec_t const& y) {
    vec_t v;
    for (size_t i = 0; i < N; ++i) {
      v(i) = std::max(x(i), y(i));
    }
    return v;
  }
};

//==============================================================================
namespace tatooine::ode::vclibs {
//==============================================================================
static constexpr inline auto rk43          = VC::odeint::RK43;
static constexpr inline auto out_of_domain = VC::odeint::OutOfDomain;
static constexpr inline auto stopped       = VC::odeint::evstate_t::Stopped;
static constexpr inline auto failed        = VC::odeint::evstate_t::Failed;
static constexpr inline auto ok            = VC::odeint::evstate_t::OK;
static constexpr inline auto abs_tol       = VC::odeint::AbsTol;
static constexpr inline auto rel_tol       = VC::odeint::RelTol;
static constexpr inline auto initial_step  = VC::odeint::InitialStep;
static constexpr inline auto max_step      = VC::odeint::MaxStep;
static constexpr inline auto max_num_steps = VC::odeint::MaxNumSteps;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, arithmetic Real, size_t N, bool B = false>
using maybe_t = typename VC::odeint::ode_t<N, Real, T, B>::maybe_vec;
template <arithmetic Real, size_t N, bool B = false>
using maybe_vec_t = maybe_t<vec<Real, N>, Real, N, B>;
template <arithmetic Real, size_t N, bool B = false>
using maybe_real_t = maybe_t<Real, Real, N, B>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, typename Real, size_t N>
concept vc_stepper_evaluator =
    std::regular_invocable<F, vec<Real, N>, Real>&& std::is_same_v<
        maybe_vec_t<Real, N>, std::invoke_result_t<F, vec<Real, N>, Real>>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
struct rungekutta43 : solver<rungekutta43<Real, N>, Real, N> {
  //============================================================================
  using this_t   = rungekutta43<Real, N>;
  using parent_t = solver<this_t, Real, N>;
  using typename parent_t::pos_t;
  using typename parent_t::vec_t;
  using vc_ode_t     = VC::odeint::ode_t<N, Real, vec_t, false>;
  using vc_stepper_t = typename vc_ode_t::template solver_t<
      VC::odeint::steppers::rk43_t<pos_t, Real>>;
  using vc_options_t = typename vc_ode_t::options_t;
  using maybe_vec    = typename vc_ode_t::maybe_vec;
  //============================================================================
 private:
  mutable vc_stepper_t m_stepper;
  //============================================================================
 public:
  constexpr rungekutta43()
      : m_stepper{vc_ode_t::solver(
            rk43, vc_options_t{abs_tol = 1e-5, rel_tol = 1e-7,
                               initial_step = 0 /*, max_step = 0.1*/})} {}
  constexpr rungekutta43(rungekutta43 const& /*other*/)
      : m_stepper{vc_ode_t::solver(
            rk43, vc_options_t{abs_tol = 1e-5, rel_tol = 1e-7,
                               initial_step = 0 /*, max_step = 0.1*/})} {}
  constexpr rungekutta43(rungekutta43&& /*other*/)
      : m_stepper{vc_ode_t::solver(
            rk43, vc_options_t{abs_tol = 1e-5, rel_tol = 1e-7,
                               initial_step = 0 /*, max_step = 0.1*/})} {}
  constexpr auto operator=(rungekutta43 const & /*other*/) -> rungekutta43& {
    return *this;
  }
  constexpr auto operator=(rungekutta43&& /*other*/) noexcept -> rungekutta43& {
    return *this;
  }
  //----------------------------------------------------------------------------
  template <typename... Options>
  rungekutta43(Options&&... options)
      : m_stepper{vc_ode_t::solver(
            rk43, vc_options_t{std::forward<Options>(options)...})} {}
  //============================================================================
  /// Continues integration of integral.
  /// if tau > 0 than it integrates forward and pushes new samples back
  /// otherwise pushes samples to front.
  template <arithmetic VReal, arithmetic Y0Real, arithmetic T0Real,
            arithmetic                          TauReal,
            stepper_callback_invocable<Real, N> StepperCallback>
  constexpr void solve(polymorphic::vectorfield<VReal, N> const& v,
                       vec<Y0Real, N> const& y0, T0Real const t0, TauReal tau,
                       StepperCallback&& callback) const {
    // do not start integration if y0, t0 is not in domain of vectorfield
    if (v(y0, t0).isnan()) {
      return;
    }

    auto dy = [&v](pos_t const& y, Real const t) -> maybe_vec {
      if (v(y, t).isnan()) {
        return out_of_domain;
      }
      return v(y, t);
    };
    solve(dy, y0, t0, tau, std::forward<StepperCallback>(callback));
  }
  //----------------------------------------------------------------------------
  template <arithmetic Y0Real, arithmetic T0Real, arithmetic TauReal,
            stepper_evaluator<Real, N>          Evaluator,
            stepper_callback_invocable<Real, N> StepperCallback>
  constexpr void solve(Evaluator&& evaluator, vec<Y0Real, N> const& y0,
                       T0Real const t0, TauReal tau,
                       StepperCallback&& callback) const {
    auto dy = [&evaluator](Real t, pos_t const& y) -> maybe_vec {
      if (auto const s = evaluator(y, t); s) {
        return *s;
      } else {
        return out_of_domain;
      }
    };
    solve(dy, y0, t0, tau, std::forward<StepperCallback>(callback));
  }
  //----------------------------------------------------------------------------
  template <arithmetic Y0Real, arithmetic T0Real, arithmetic TauReal,
            vc_stepper_evaluator<Real, N>       Evaluator,
            stepper_callback_invocable<Real, N> StepperCallback>
  constexpr void solve(Evaluator&& evaluator, vec<Y0Real, N> const& y0,
                       T0Real const t0, TauReal tau,
                       StepperCallback&& callback) const {
    constexpr auto callback_takes_derivative =
        std::is_invocable_v<StepperCallback, pos_t, Real, vec_t>;

    if (tau == 0) {
      return;
    }
    // do not start integration if y0, t0 is not in domain of vectorfield

    auto dy = [&evaluator](Real t, pos_t const& y) {
      return evaluator(y, t);
    };

    m_stepper.initialize(dy, t0, t0 + tau, y0);
    auto wrapped_callback = [&] {
      if constexpr (!callback_takes_derivative) {
        return [&](Real const t, pos_t const& y) {
          callback(y, t);
        };
      } else {
        return [&](Real const t, pos_t const& y, vec_t const& dy) {
          callback(y, t, dy);
        };
      }
    }();
    m_stepper.integrate(dy,
                        vc_ode_t::Output >> vc_ode_t::sink(wrapped_callback));
  }
};
//==============================================================================
}  // namespace tatooine::ode::vclibs
//==============================================================================
#endif
