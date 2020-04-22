#ifndef TATOOINE_INTEGRATION_VCLIBS_RK43_H
#define TATOOINE_INTEGRATION_VCLIBS_RK43_H

#include <tatooine/interpolation.h>

#include <boost/range/numeric.hpp>
#include <vcode/odeint.hh>

#include "../integrator.h"

//==============================================================================
template <typename Real, size_t N>
struct VC::odeint::vector_operations_t<tatooine::vec<Real, N>> {
  using vec_t = tatooine::vec<Real, N>;

  //----------------------------------------------------------------------------
  static constexpr bool isfinitenorm(const vec_t& v) {
    for (auto c : v) {
      if (!std::isfinite(c)) { return false; }
    }
    return true;
  }

  //----------------------------------------------------------------------------
  static constexpr Real norm2(const vec_t& x) {
    static_assert(std::is_floating_point<Real>(), "require floating point");
    return std::sqrt(sqr(x));
  }

  //----------------------------------------------------------------------------
  static constexpr Real norm1(const vec_t& x) { return tatooine::norm1(x); }
  //----------------------------------------------------------------------------
  static constexpr Real norminf(const vec_t& x) { return norm_inf(x); }
  //----------------------------------------------------------------------------
  static constexpr auto abs(vec_t v) {
    for (size_t i = 0; i < N; ++i) { v(i) = std::abs(v(i)); }
    return v;
  }

  //----------------------------------------------------------------------------
  static constexpr auto max(const vec_t& x, const vec_t& y) {
    vec_t v;
    for (size_t i = 0; i < N; ++i) { v(i) = std::max(x(i), y(i)); }
    return v;
  }
};

//==============================================================================
namespace tatooine::integration::vclibs {
//==============================================================================

static constexpr inline auto rk43          = VC::odeint::RK43;
static constexpr inline auto out_of_domain = VC::odeint::OutOfDomain;
static constexpr inline auto abs_tol       = VC::odeint::AbsTol;
static constexpr inline auto rel_tol       = VC::odeint::RelTol;
static constexpr inline auto initial_step  = VC::odeint::InitialStep;
static constexpr inline auto max_step      = VC::odeint::MaxStep;
static constexpr inline auto max_num_steps = VC::odeint::MaxNumSteps;

template <typename Real, size_t N,
          template <typename> typename InterpolationKernel =
              interpolation::hermite>
struct rungekutta43 : integrator<Real, N, InterpolationKernel,
                                 rungekutta43<Real, N, InterpolationKernel>> {
  //============================================================================
  using this_t     = rungekutta43<Real, N, InterpolationKernel>;
  using parent_t   = integrator<Real, N, InterpolationKernel, this_t>;
  using integral_t = typename parent_t::integral_t;
  using pos_t      = typename parent_t::pos_t;
  using ode_t      = VC::odeint::ode_t<2, Real, vec<Real, N>, false>;
  using options_t  = typename ode_t::options_t;

  //============================================================================
 private:
  options_t m_options;

  //============================================================================
 public:
  template <typename V>
  rungekutta43(const vectorfield<V, Real, N>&)
      : m_options{abs_tol = 1e-4, rel_tol = 1e-4, initial_step = 0,
                  max_step = 0.1} {}
  rungekutta43()
      : m_options{abs_tol = 1e-4, rel_tol = 1e-4, initial_step = 0,
                  max_step = 0.1} {}
  rungekutta43(const rungekutta43& other)
      : parent_t{other},
        m_options{abs_tol = 1e-4, rel_tol = 1e-4, initial_step = 0,
                  max_step = 0.1} {}
  rungekutta43(rungekutta43&& other) noexcept
      : parent_t{std::move(other)},
        m_options{abs_tol = 1e-4, rel_tol = 1e-4, initial_step = 0,
                  max_step = 0.1} {}
  //----------------------------------------------------------------------------
  template <typename... Options,
            std::enable_if_t<!(is_field_v<Options> && ...), bool> = true>
  rungekutta43(Options&&... options)
      : m_options{std::forward<Options>(options)...} {}

  //============================================================================
 private:
  friend parent_t;

  //----------------------------------------------------------------------------
  template <typename V>
  void calc_forward(const V& v, integral_t& integral, const pos_t& y0, Real t0,
                    Real tau) const {
    // do not start integration if y0,t0 is not in domain of vectorfield
    if (!v.in_domain(y0, t0)) { return; }

    auto dy = [&v](Real t, const pos_t& y) -> typename ode_t::maybe_vec {
      if (!v.in_domain(y, t)) { return VC::odeint::OutOfDomain; }
      return v(y, t);
    };
    auto stepper = ode_t::solver(rk43, m_options);
    stepper.initialize(dy, t0, t0 + tau, y0);

    auto& tangents = integral.tangents_property();
    stepper.integrate(
        dy, ode_t::Output >>
                ode_t::sink([&v, &integral, &tangents](auto t, const auto& y,
                                                   const auto& dy) {
                  //if (integral.num_vertices() > 0 &&
                  //    std::abs(integral.back_parameterization() - t) < 1e-13) {
                  //  return;
                  //}
                  integral.push_back(y, t, false);
                  tangents.back() = dy;
                }));
  }
  //----------------------------------------------------------------------------
  template <typename V>
  void calc_backward(const V& v, integral_t& integral, const pos_t& y0, Real t0,
                     Real tau) const {
    // do not start integration if y0,t0 is not in domain of vectorfield
    if (!v.in_domain(y0, t0)) { return; }

    auto dy = [&v](Real t, const pos_t& y) -> typename ode_t::maybe_vec {
      if (!v.in_domain(y, t)) { return VC::odeint::OutOfDomain; }
      return v(y, t);
    };
    auto stepper = ode_t::solver(rk43, m_options);
    stepper.initialize(dy, t0, t0 + tau, y0);

    auto& tangents = integral.tangents_property();
    stepper.integrate(
        dy, ode_t::Output >>
                ode_t::sink([&v, &integral, &tangents](auto t, const auto& y,
                                                       const auto& dy) {
                  if (integral.num_vertices() > 0 &&
                      std::abs(integral.front_parameterization() - t) < 1e-13) {
                    return;
                  }
                  integral.push_front(y, t, false);
                  tangents.front() = dy;
                }));
  }
};

template <typename V, typename Real, size_t N>
rungekutta43(const vectorfield<V, Real, N>&)
    ->rungekutta43<Real, N, interpolation::hermite>;
//==============================================================================
}  // namespace tatooine::integration::vclibs
//==============================================================================

#endif
