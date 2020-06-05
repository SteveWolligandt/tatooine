#ifndef TATOOINE_ODE_ODE_SOLVER_H
#define TATOOINE_ODE_ODE_SOLVER_H
//==============================================================================
#include <cassert>
#include <tatooine/concepts.h>
#include <map>
#include "../cache.h"
#include "../field.h"
#include "../line.h"
//==============================================================================
namespace tatooine::integration {
//==============================================================================
template <typename Derived>
struct ode_solver : crtp<Derived> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using parent_t   = crtp<Derived>;
  using parent_t::as_derived;

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 private:
  //----------------------------------------------------------------------------
  template <template <typename>
            typename InterpolationKernel = interpolation::hermite,
            size_t N, typename V, std::floating_point VReal, typename Integral,
            arithmetic Y0Real, arithmetic T0Real, arithmetic TauReal>
  auto solve(const vectorfield<V, VReal, N>& v, vec<Y0Real, N>& y0,
             T0Real t0, TauReal tau) const {
    using real_t = promote_t<VReal, Y0Real, T0Real, TauReal>;
    parameterized_line<real_t, N, InterpolationKernel> integral_curve;
    integral_curve.push_back(std::move(y0), t0);
    solve(v, integral_curve, tau);
    return integral_curve;
  }
  //----------------------------------------------------------------------------
  template <template <typename>
            typename InterpolationKernel = interpolation::hermite,
            size_t N, typename V, std::floating_point VReal, typename Integral,
            arithmetic Y0Real, arithmetic T0Real, arithmetic BTauReal,
            arithmetic FTauReal>
  auto solve(const vectorfield<V, VReal, N> &v, vec<Y0Real, N> &y0, T0Real t0,
             TauReal btau, TauReal ftau) const {
    auto integral_curve = solve(v, y0, t0, btau);
    as_derived().solve(v, integral_curve, ftau);
    return integral_curve;
  }
  //----------------------------------------------------------------------------
  /// Continues integration of integral.
  /// if tau > 0 than it integrates forward and pushes new samples back
  /// otherwise pushes samples to front.
  template <size_t N, typename V, std::floating_point VReal, typename Integral,
            arithmetic X0Real, arithmetic T0Real, arithmetic CurveReal,
            template <typename> typename InterpolationKernel,
            arithmetic TauReal>
  void solve(
      const vectorfield<V, VReal, N>&                        v,
      parameterized_line<CurveReal, N, InterpolationKernel>& integral_curve,
      TauReal                                                tau) const {
    as_derived().solve(v, integral, tau);
  }
  //----------------------------------------------------------------------------
  template <size_t N, typename V, std::floating_point VReal, typename Integral,
            arithmetic X0Real, arithmetic T0Real, arithmetic CurveReal,
            template <typename> typename InterpolationKernel,
            arithmetic BTauReal, arithmetic FTauReal>
  void solve(
      const vectorfield<V, VReal, N> &                       v,
      parameterized_line<CurveReal, N, InterpolationKernel> &integral_curve,
      BTauReal btau, FTauReal ftau) const {
    as_derived().solve(v, integral, btau);
    as_derived().solve(v, integral, ftau);
  }

 public:
  //----------------------------------------------------------------------------
  template <typename V>
  const auto &integrate(const V &v, const pos_t &y0, Real t0, Real tau) const {
    auto [it, new_integral] = m_cache.emplace({t0, y0});
    auto &integral            = it->second;

    auto &[backward_on_border, forward_on_border] =
        m_on_domain_border[{y0, t0}];

    // integral not yet integrated
    if (new_integral || integral.empty()) {
      integral = integrate_uncached(v, y0, t0, tau, backward_on_border,
                                    forward_on_border);
    }

    // integral has to be integrated further
    else {
      if (tau > 0 && integral.back_parameterization() < t0 + tau &&
          !forward_on_border) {
        continue_forward(v, integral, tau, forward_on_border);
      } else if (tau < 0 && integral.front_parameterization() > t0 + tau &&
                 !backward_on_border) {
        continue_backward(v, integral, tau, backward_on_border);
      }
    }
    return integral;
  }
  //----------------------------------------------------------------------------
  template <typename V>
  const auto &integrate(const V &v, const pos_t &y0, Real t0, Real backward_tau,
                        Real forward_tau) const {
    auto [it, new_integral] = m_cache.emplace(std::pair{t0, y0});
    auto &integral          = it->second;
    assert(backward_tau <= 0);
    assert(forward_tau >= 0);

    auto &[backward_on_border, forward_on_border] =
        m_on_domain_border[{y0, t0}];

    // integral not yet integrated
    if (new_integral || integral.empty()) {
      integral = integrate_uncached(v, y0, t0, backward_tau, forward_tau,
                                    backward_on_border, forward_on_border);

      // integral has to be integrated further
    } else {
      if (backward_tau < 0 &&
          integral.front_parameterization() > t0 + backward_tau &&
          !backward_on_border) {
        continue_backward(v, integral, backward_tau, backward_on_border);
      }

      if (forward_tau > 0 &&
          integral.back_parameterization() < t0 + forward_tau &&
          !forward_on_border) {
        continue_forward(v, integral, forward_tau, forward_on_border);
      }
    }
    return integral;
  }
  //----------------------------------------------------------------------------
  template <typename V>
  auto integrate_uncached(const V &v, const pos_t &y0, Real t0,
                          Real tau) const {
    integral_t integral;
    if (tau < 0) {
      calc_backward(v, integral, y0, t0, tau);
    } else {
      calc_forward(v, integral, y0, t0, tau);
    }
    integral.update_interpolators();
    return integral;
  }
  //----------------------------------------------------------------------------
  template <typename V>
  auto integrate_uncached(const V &v, const pos_t &y0, Real t0,
                          Real btau, Real ftau) const {
    integral_t integral;
    calc_backward(v, integral, y0, t0, btau);
    calc_forward(v, integral, y0, t0, ftau);
    integral.update_interpolators();
    return integral;
  }
  //----------------------------------------------------------------------------
  template <typename V>
  auto integrate_uncached(const V &v, const pos_t &y0, Real t0, Real tau,
                          bool &backward_on_border,
                          bool &forward_on_border) const {
    integral_t integral;
    if (tau < 0) {
      calc_backward(v, integral, y0, t0, tau);
    } else {
      calc_forward(v, integral, y0, t0, tau);
    }
    if (!integral.empty()) {
      if (tau < 0 &&
          std::abs(integral.front_parameterization() - (t0 + tau)) > 1e-5)
        backward_on_border = true;
      else if (tau > 0 && std::abs(integral.back_parameterization() - (t0 + tau)) > 1e-5)
        forward_on_border = true;
    } else {
      backward_on_border = forward_on_border = true;
    }
    integral.update_interpolators();
    return integral;
  }
  //----------------------------------------------------------------------------
  template <typename V>
  auto integrate_uncached(const V &v, const pos_t &y0, Real t0,
                          Real backward_tau, Real forward_tau,
                          bool &backward_on_border,
                          bool &forward_on_border) const {
    integral_t integral;
    calc_backward(v, integral, y0, t0, backward_tau);
    calc_forward(v, integral, y0, t0, forward_tau);

    if (!integral.empty()) {
      if (std::abs(integral.front_parameterization() - (t0 + backward_tau)) >
          1e-7) {
        backward_on_border = true;
      }
      if (std::abs(integral.back_parameterization() - (t0 + forward_tau)) >
          1e-7) {
        forward_on_border = true;
      }
    } else {
      backward_on_border = forward_on_border = true;
    }
    integral.update_interpolators();
    return integral;
  }
  //----------------------------------------------------------------------------
  /// continues integration of integral
  template <typename V>
  auto &continue_forward(const V &v, integral_t &integral, const Real tau,
                         bool &forward_on_border) const {
    const Real & t0 = integral.back_parameterization();
    const pos_t &y0 = integral.back_vertex();

    Real tau_rest = t0 + tau - integral.back_parameterization();
    calc_forward(v, integral, y0, t0, tau_rest);
    if (!integral.empty() && integral.back_parameterization() < t0 + tau_rest) {
      forward_on_border = true;
    }
    integral.update_interpolators();
    return integral;
  }
  //----------------------------------------------------------------------------
  /// continues integration of integral
  template <typename V>
  auto &continue_backward(const V &v, integral_t &integral, Real tau,
                          bool &backward_on_border) const {
    const Real & t0 = integral.front_parameterization();
    const pos_t &y0 = integral.front_vertex();

    Real tau_rest = t0 + tau - integral.front_parameterization();
    calc_backward(v, integral, y0, t0, tau_rest);
    if (!integral.empty() &&
        integral.front_parameterization() > t0 + tau_rest) {
      backward_on_border = true;
    }
    integral.update_interpolators();
    return integral;
  }

  auto &      cache() { return m_cache; }
  const auto &cache() const { return m_cache; }
};
//==============================================================================
}  // namespace tatooine::integration
//==============================================================================
#endif
