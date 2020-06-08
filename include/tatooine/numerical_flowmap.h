#ifndef TATOOINE_NUMERICAL_FLOWMAP_H
#define TATOOINE_NUMERICAL_FLOWMAP_H
//==============================================================================
#include "exceptions.h"
#include "field.h"
#include "interpolation.h"
#include "ode/vclibs/rungekutta43.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, template <typename, size_t> typename ODESolver,
          template <typename> typename InterpolationKernel>
struct numerical_flowmap {
  using this_t = numerical_flowmap<V, ODESolver, InterpolationKernel>;
  using real_t = typename V::real_t;
  static constexpr auto num_dimensions() { return V::num_dimensions(); }
  using pos_t = vec<real_t, num_dimensions()>;
  using integral_curve_t = parameterized_line<real_t, num_dimensions(), InterpolationKernel>;
  using cache_t = tatooine::cache<std::pair<real_t, pos_t>, integral_curve_t>;
  using ode_solver_t = ODESolver<real_t, num_dimensions()>;
  //============================================================================
 private:
  V const&        m_v;
  ode_solver_t    m_ode_solver;
  mutable cache_t m_cache;
  mutable std::map<std::pair<pos_t, real_t>, std::pair<bool, bool>>
      m_on_domain_border;
  //============================================================================
 public:
  template <typename VReal, size_t N>
  constexpr numerical_flowmap(vectorfield<V, VReal, N> const& v)
      : m_v{v.as_derived()} {}
  //----------------------------------------------------------------------------
  template <typename VReal, size_t N>
  constexpr numerical_flowmap(vectorfield<V, VReal, N> const& v,
                              ode_solver_t const&             ode_solver)
      : m_v{v.as_derived()}, m_ode_solver{ode_solver} {}
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(pos_t const& y0, real_t const t0,
                                        real_t tau) const -> pos_t {
    if (tau == 0) { return y0; }
    auto const& integral_curve = cached_curve(y0, t0, tau);
    if ((tau < 0 && t0 + tau < integral_curve.front_parameterization()) ||
        (tau > 0 && t0 + tau > integral_curve.back_parameterization())) {
      throw out_of_domain_error{};
    }
    return integral_curve(t0 + tau);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator()(pos_t const& y0, real_t const t0,
                                          real_t tau) const -> pos_t {
    return evaluate(y0, t0, tau);
  }
  //----------------------------------------------------------------------------
  auto integral_curve(pos_t const& y0, real_t const t0, real_t const tau) const {
    integral_curve_t c;
    c.push_back(y0, t0);
    auto const full_integration = continue_integration(c, tau);
    return std::pair{std::move(c), full_integration};
  }
  //----------------------------------------------------------------------------
  auto integral_curve(pos_t const& y0, real_t const t0, real_t const btau,
                      real_t const ftau) const {
    integral_curve_t c;
    c.push_back(y0, t0);

    bool const backward_full = [this, &c, btau] {
      if (btau < 0) { return continue_integration(c, btau); }
      return true;
    }();
    bool const forward_full = [this, &c, ftau] {
      if (ftau > 0) { return continue_integration(c, ftau); }
      return true;
    }();
    return std::tuple{std::move(c), backward_full, forward_full};
  }
  //----------------------------------------------------------------------------
  /// Continues integration if integral_curve.
  /// If tau > 0 it takes front of integral_curve as start position and time.
  /// If tau < 0 it takes back of integral_curve as start position and time.
  /// \return true if could integrate all tau, false if hit domain border or
  /// something else went wrong.
  bool continue_integration(integral_curve_t& integral_curve,
                            real_t            tau) const {
    auto& tangents = integral_curve.tangents_property();

    auto const& y0 = [&integral_curve, tau] {
      if (tau > 0) { return integral_curve.back_vertex(); }
      return integral_curve.front_vertex();
    }();
    auto const& t0 = [&integral_curve, tau] {
      if (tau > 0) { return integral_curve.back_parameterization(); }
      return integral_curve.front_parameterization();
    }();
    auto callback = [this, &integral_curve, &tangents, tau](
                        auto t, const auto& y, const auto& dy) {
       if (integral_curve.num_vertices() > 0 &&
          std::abs(integral_curve.back_parameterization() - t) < 1e-13) {
        return;
      }
      if (tau < 0) {
        integral_curve.push_front(y, t, false);
        tangents.front() = dy;
      } else {
        integral_curve.push_back(y, t, false);
        tangents.back() = dy;
      }
    };

    m_ode_solver.solve(m_v, y0, t0, tau, callback);
    integral_curve.update_interpolators();
    if (!integral_curve.empty()) {
      if ((tau > 0 && integral_curve.back_parameterization() < t0 + tau) ||
          (tau < 0 && integral_curve.front_parameterization() < t0 + tau)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto cached_curve(pos_t const& y0, real_t const t0) const -> auto const& {
    return *m_cache.emplace({t0, y0}).first->second;
  }
  //----------------------------------------------------------------------------
  auto cached_curve(pos_t const& y0, real_t const t0, real_t const tau) const
      -> auto const& {
    return cached_curve(y0, t0, tau < 0 ? tau : 0, tau > 0 ? tau : 0);
  }
  //----------------------------------------------------------------------------
  auto cached_curve(pos_t const& y0, real_t const t0, real_t btau,
                    real_t ftau) const -> auto const& {
    auto [curve_it, new_integral_curve] = m_cache.emplace({t0, y0});
    auto& curve                         = curve_it->second;

    auto& [backward_on_border, forward_on_border] =
        m_on_domain_border[{y0, t0}];

    if (new_integral_curve || curve.empty()) {
      // integral_curve not yet integrated
      auto [fresh_curve, fullback, fullforw] =
          integral_curve(y0, t0, btau, ftau);
      curve = std::move(fresh_curve);
      backward_on_border = !fullback;
      forward_on_border  = !fullforw;
    } else {
      if (btau > ftau) { std::swap(btau, ftau); }
      if (auto const tf = curve.front_parameterization();
          btau < 0 && tf > t0 + btau && !backward_on_border) {
        // continue integration in backward time
        bool const full    = continue_integration(curve, t0 + btau - tf);
        backward_on_border = !full;
      }
      if (auto const tb = curve.back_parameterization();
          ftau > 0 && tb < t0 + ftau && !forward_on_border) {
        // continue integration in forward time
        bool const full   = continue_integration(curve, t0 + ftau - tb);
        forward_on_border = !full;
      }
    }
    return curve;
  }
  //============================================================================
  auto vectorfield() const -> auto const& { return m_v; }
  auto vectorfield() -> auto& { return m_v; }
  //----------------------------------------------------------------------------
  auto ode_solver() const -> auto const& { return *m_ode_solver; }
  auto ode_solver() -> auto& { return *m_ode_solver; }
};
//==============================================================================
template <typename V, typename Real, size_t N>
numerical_flowmap(vectorfield<V, Real, N> const&)
    -> numerical_flowmap<V, ode::vclibs::rungekutta43, interpolation::hermite>;
//==============================================================================
template <typename V, typename VReal, size_t N>
auto flowmap(vectorfield<V, VReal, N> const& v) {
  return numerical_flowmap<V, ode::vclibs::rungekutta43,
                           interpolation::hermite>{v};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
