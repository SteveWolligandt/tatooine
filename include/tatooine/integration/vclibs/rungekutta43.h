#ifndef TATOOINE_INTEGRATION_VCLIBS_RK43_H
#define TATOOINE_INTEGRATION_VCLIBS_RK43_H

#include <boost/range/numeric.hpp>
#include <vcode/odeint.hh>
#include "../integrator.h"

//==============================================================================
template <typename Real, size_t N>
struct VC::odeint::vector_operations_t<tatooine::tensor<Real, N>> {
  using vec_t = tatooine::tensor<Real, N>;

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
  static constexpr Real norm1(const vec_t& x) {
    Real norm = 0;
    for (size_t i = 0; i < N; ++i) {
      norm += std::abs(x(i));
    }
    return norm;
  }

  //----------------------------------------------------------------------------
  static constexpr Real norminf(const vec_t& x) {
    Real norm = -std::numeric_limits<Real>::max();
    for (size_t i = 0; i < N; ++i) {
      norm = std::max(norm, std::abs(x(i)));
    }
    return norm;
  }

  //----------------------------------------------------------------------------
  static constexpr auto abs(vec_t v) {
    for (size_t i = 0; i < N; ++i) {
      v(i) = std::abs(v(i));
    }
    return v;
  }

  //----------------------------------------------------------------------------
  static constexpr auto max(const vec_t& x, const vec_t& y) {
    vec_t v;
    for (size_t i = 0; i < N; ++i) {
      v(i) = std::max(x(i), y(i));
    }
    return v;
  }
};

//==============================================================================
namespace tatooine::integration::vclibs {
//==============================================================================

template <typename Real, size_t N>
struct rungekutta43 : integrator<Real, N, rungekutta43<Real, N>> {
  //============================================================================
  using this_t     = rungekutta43<Real, N>;
  using parent_t   = integrator<Real, N, this_t>;
  using integral_t = typename parent_t::integral_t;
  using pos_t      = typename parent_t::pos_t;
  using ode_t      = VC::odeint::ode_t<2, Real, tensor<Real, N>, false>;
  using options_t  = typename ode_t::options_t;

  static constexpr auto RK43        = VC::odeint::RK43;
  static constexpr auto OutOfDomain = VC::odeint::OutOfDomain;
  static constexpr auto AbsTol      = VC::odeint::AbsTol;
  static constexpr auto RelTol      = VC::odeint::RelTol;
  static constexpr auto InitialStep = VC::odeint::InitialStep;
  static constexpr auto MaxStep     = VC::odeint::MaxStep;
  static constexpr auto MaxNumSteps = VC::odeint::MaxNumSteps;

  //============================================================================
 private:
  options_t m_options;

  //============================================================================
 public:
  rungekutta43()
      : m_options{AbsTol = 1e-4, RelTol = 1e-4, InitialStep = 0,
                  MaxStep = 0.01} {}
  rungekutta43(const rungekutta43&)            = default;
  rungekutta43(rungekutta43&&)                 = default;
  rungekutta43& operator=(const rungekutta43&) = default;
  rungekutta43& operator=(rungekutta43&&)      = default;
  //----------------------------------------------------------------------------
  template <typename... Options>
  rungekutta43(Options&&... options)
      : m_options{std::forward<Options>(options)...} {}

  //============================================================================
 private:
  friend parent_t;

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto& calc_forward(const vf_t& vf, integral_t& integral, const pos_t& y0, Real t0,
             Real tau) const {
    // do not start integration if y0,t0 is not in domain of vectorfield
    if (!vf.in_domain(y0, t0)) { return integral; }

    auto dy = [&vf](Real t, const pos_t& y) -> typename ode_t::maybe_vec {
      if (!vf.in_domain(y, t)) return VC::odeint::OutOfDomain;
      return vf(y, t);
    };
    auto stepper = ode_t::solver(RK43, m_options);
    stepper.initialize(dy, t0, t0 + tau, y0);

    stepper.integrate(
        dy, ode_t::Output >> ode_t::sink([&integral](auto t, const auto& y) {
              bool use = true;
              if (!integral.empty() &&
                  distance(integral.back_position(), y) < 1e-6) {
                use = false;
              }
              if (use) { integral.push_back(y, t); }
            }));
    return integral;
  }

  //----------------------------------------------------------------------------
  template <typename vf_t>
  auto& calc_backward(const vf_t& vf, integral_t& integral, const pos_t& y0, Real t0,
             Real tau) const {
    // do not start integration if y0,t0 is not in domain of vectorfield
    if (!vf.in_domain(y0, t0)) { return integral; }

    auto dy = [&vf](Real t, const pos_t& y) -> typename ode_t::maybe_vec {
      if (!vf.in_domain(y, t)) return VC::odeint::OutOfDomain;
      return vf(y, t);
    };
    auto stepper = ode_t::solver(RK43, m_options);
    stepper.initialize(dy, t0, t0 + tau, y0);

    stepper.integrate(
        dy, ode_t::Output >> ode_t::sink([&integral](auto t, const auto& y) {
              bool use = true;
              if (!integral.empty() &&
                  distance(integral.back_position(), y) < 1e-6) {
                use = false;
              }
              if (use) { integral.push_front(y, t); }
            }));
    return integral;
  }
};

//==============================================================================
}  // namespace tatooine::integration::vclibs
//==============================================================================

#endif
