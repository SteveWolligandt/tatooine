#ifndef TATOOINE_INTEGRATION_VCLIBS_RK43_H
#define TATOOINE_INTEGRATION_VCLIBS_RK43_H

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

template <typename Real, size_t N>
struct rungekutta43 : integrator<Real, N, rungekutta43<Real, N>> {
  //============================================================================
  using this_t     = rungekutta43<Real, N>;
  using parent_t   = integrator<Real, N, this_t>;
  using integral_t = typename parent_t::integral_t;
  using pos_t      = typename parent_t::pos_t;
  using ode_t      = VC::odeint::ode_t<2, Real, vec<Real, N>, false>;
  using options_t  = typename ode_t::options_t;


  //============================================================================
 private:
  options_t m_options;

  //============================================================================
 public:
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
  template <typename... Options>
  rungekutta43(Options&&... options)
      : m_options{std::forward<Options>(options)...} {}

  //============================================================================
 private:
  friend parent_t;

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto& calc_forward(const vf_t& vf, integral_t& integral, const pos_t& y0,
                     Real t0, Real tau) const {
    // do not start integration if y0,t0 is not in domain of vectorfield
    if (!vf.in_domain(y0, t0)) { return integral; }

    auto dy = [&vf](Real t, const pos_t& y) -> typename ode_t::maybe_vec {
      if (!vf.in_domain(y, t)) return VC::odeint::OutOfDomain;
      return vf(y, t);
    };
    auto stepper = ode_t::solver(rk43, m_options);
    stepper.initialize(dy, t0, t0 + tau, y0);

    auto& tangents = integral.tangents_property();
    stepper.integrate(
        dy, ode_t::Output >> ode_t::sink([&integral, &tangents](
                                             auto t, const auto& y,
                                             const auto& dy) {
              bool use = true;
              if (!integral.empty() &&
                  distance(integral.back_vertex(), y) < 1e-6) {
                use = false;
              }
              if (use) {
                integral.push_back(y, t);
                tangents.back() = dy;
              }
            }));
    return integral;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto& calc_backward(const vf_t& vf, integral_t& integral, const pos_t& y0,
                      Real t0, Real tau) const {
    // do not start integration if y0,t0 is not in domain of vectorfield
    if (!vf.in_domain(y0, t0)) { return integral; }

    auto dy = [&vf](Real t, const pos_t& y) -> typename ode_t::maybe_vec {
      if (!vf.in_domain(y, t)) return VC::odeint::OutOfDomain;
      return vf(y, t);
    };
    auto stepper = ode_t::solver(rk43, m_options);
    stepper.initialize(dy, t0, t0 + tau, y0);

    auto& tangents = integral.tangents_property();
    stepper.integrate(
        dy, ode_t::Output >> ode_t::sink([&integral, &tangents](
                                             auto t, const auto& y,
                                             const auto& dy) {
              bool use = true;
              if (!integral.empty() &&
                  distance(integral.back_vertex(), y) < 1e-6) {
                use = false;
              }
              if (use) {
                integral.push_front(y, t);
                tangents.front() = dy;
              }
            }));
    return integral;
  }
};

//==============================================================================
}  // namespace tatooine::integration::vclibs
//==============================================================================

#endif
