#ifndef TATOOINE_INTEGRATION_INTEGRATOR_H
#define TATOOINE_INTEGRATION_INTEGRATOR_H

#include <cassert>
#include <map>
#include "../cache.h"
#include "../field.h"
#include "../line.h"

//==============================================================================
namespace tatooine::integration {
//==============================================================================

template <typename Real, size_t N,
          template <typename> typename InterpolationKernel, typename Derived>
struct integrator : crtp<Derived> {
  using real_t     = Real;
  using parent_t   = crtp<Derived>;
  using integral_t = parameterized_line<Real, N, InterpolationKernel>;
  using pos_t      = vec<Real, N>;
  using cache_t    = cache<std::pair<Real, pos_t>, integral_t>;

  using parent_t::as_derived;

 private:
  mutable cache_t m_cache;
  mutable std::map<std::pair<pos_t, Real>, std::pair<bool, bool>>
      m_on_domain_border;

  //----------------------------------------------------------------------------
  template <typename V>
  auto calc_forward(const V &v, integral_t &integral, const pos_t &y0, Real t0,
                    Real tau) const {
    return as_derived().calc_forward(v, integral, y0, t0, tau);
  }

  //----------------------------------------------------------------------------
  template <typename V>
  auto calc_backward(const V &v, integral_t &integral, const pos_t &y0, Real t0,
                     Real tau) const {
    return as_derived().calc_backward(v, integral, y0, t0, tau);
  }

 public:
  integrator()                       = default;
  integrator(const integrator &)     = default;
  //integrator(integrator &&) noexcept = default;
  integrator &operator=(const integrator &) = default;
  //integrator &operator=(integrator &&) noexcept = default;
  //----------------------------------------------------------------------------
  template <typename V>
  const auto &integrate(const V &v, const pos_t &y0, Real t0, Real tau) const {
    auto [it, new_streamline] = m_cache.emplace({t0, y0});
    auto &integral            = it->second;

    auto &[backward_on_border, forward_on_border] =
        m_on_domain_border[{y0, t0}];

    // integral not yet integrated
    if (new_streamline || integral.empty()) {
      integral = integrate_uncached(v, y0, t0, tau, backward_on_border,
                                    forward_on_border);
    }

    // integral has to be integrated a bit longer
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
    auto [it, new_streamline] = m_cache.emplace(std::pair{t0, y0});
    auto &integral            = it->second;
    assert(backward_tau <= 0);
    assert(forward_tau >= 0);

    auto &[backward_on_border, forward_on_border] =
        m_on_domain_border[{y0, t0}];

    // integral not yet integrated
    if (new_streamline || integral.empty())
      integral = integrate_uncached(v, y0, t0, backward_tau, forward_tau,
                                    backward_on_border, forward_on_border);

    // integral has to be integrated a bit longer
    else {
      if (backward_tau < 0 &&
          integral.front_parameterization() > t0 + backward_tau &&
          !backward_on_border)
        continue_backward(v, integral, backward_tau, backward_on_border);

      if (forward_tau > 0 &&
          integral.back_parameterization() < t0 + forward_tau &&
          !forward_on_border)
        continue_forward(v, integral, forward_tau, forward_on_border);
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
      if (tau < 0 && integral.front_parameterization() > t0 + tau)
        backward_on_border = true;
      else if (tau > 0 && integral.back_parameterization() < t0 + tau)
        forward_on_border = true;
    } else
      backward_on_border = forward_on_border = true;
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
      if (integral.front_parameterization() > t0 + backward_tau)
        backward_on_border = true;
      if (integral.back_parameterization() < t0 + forward_tau)
        forward_on_border = true;
    } else
      backward_on_border = forward_on_border = true;
    return integral;
  }

  //----------------------------------------------------------------------------
  //! continues integration of integral
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
    return integral;
  }

  //----------------------------------------------------------------------------
  //! continues integration of integral
  template <typename V>
  auto &continue_backward(const V &v, integral_t &integral, Real tau,
                          bool &backward_on_border) const {
    const Real & t0 = integral.front_parameterization();
    const pos_t &y0 = integral.front_vertex();

    Real tau_rest = t0 + tau - integral.front_parameterization();
    auto cont     = calc_backward(v, integral, y0, t0, tau_rest);
    if (!integral.empty() &&
        integral.front_parameterization() > t0 + tau_rest) {
      backward_on_border = true;
    }
    return integral;
  }

  auto &      cache() { return m_cache; }
  const auto &cache() const { return m_cache; }
};

//==============================================================================
}  // namespace tatooine::integration
//==============================================================================

#endif
