#ifndef TATOOINE_FLOWMAP_H
#define TATOOINE_FLOWMAP_H
//==============================================================================
#include "field.h"
#include "interpolation.h"
#include "ode/vclibs/rungekutta43.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V,
          template <typename, size_t, template <typename> typename>
          typename ODESolver = ode::vclibs::rungekutta43,
          template <typename> typename InterpolationKernel =interpolation::hermite>
struct flowmap {
  using this_t = flowmap<V, ODESolver, InterpolationKernel>;
  using parent_t::num_dimensions;
  using integral_curve_t = parameterized_line<Real, N, InterpolationKernel>;
  using cache_t = tatooine::cache<std::pair<Real, pos_t>, integral_curve_t>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  using parent_t::operator();
  using ode_solver_t =
      ODESolver<real_t, parent_t::num_dimensions(), InterpolationKernel>;
  //============================================================================
 private:
  V const&        m_vectorfield;
  ode_solver_t    m_ode_solver;
  mutable cache_t m_cache;
  mutable std::map<std::pair<pos_t, Real>, std::pair<bool, bool>>
      m_on_domain_border;
  //============================================================================
 public:
  template <typename FieldReal, typename TauReal, size_t N>
  constexpr flowmap(const vectorfield<V, FieldReal, N>& vf,
                    const ode_solver_t& ode_solver, TauReal tau)
      : m_vectorfield{vf.as_derived()},
        m_ode_solver{ode_solver} {}
  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, real_t t0, real_t tau) const {
    return m_ode_solver->integrate(m_vectorfield, x, t0, tau)(tau);
  }
  //----------------------------------------------------------------------------
  constexpr tensor_t operator()(const pos_t& x, real_t t0, real_t tau) const {
    return evaluate(x, t0, tau);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_t& x, real_t t, real_t tau) const
      -> tensor_t final {
    if (tau == 0) { return x; }
    const auto& integral = m_ode_solver->integrate(m_vectorfield, x, t, tau);
    if (integral.empty()) { return x; }
    return integral(t + tau);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto operator()(const pos_t& x, real_t t,
                                          real_t tau) const -> tensor_t final {
    return evaluate(x, t, tau);
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(const pos_t& x, real_t t) const
      -> bool final {
    return m_vectorfield.in_domain(x, t);
  }
  //----------------------------------------------------------------------------
  auto integral_curve() const {
    integral_curve_t integral_curve;
    m_ode_solver.solve();
  }
  //----------------------------------------------------------------------------
  auto cached_curve(pos_t const& y0, real_t const t0, real_t const tau) const
      -> integral_curve_t const& {
    auto [it, new_integral_curve] = m_cache.emplace({t0, y0});
    auto &curve            = it->second;

    auto &[backward_on_border, forward_on_border] =
        m_on_domain_border[{y0, t0}];

    if (new_integral_curve || curve.empty()) {
    // integral_curve not yet integrated
      curve = integral_curve(v, y0, t0, tau, backward_on_border,
                                    forward_on_border);
    } else {
      // integral_curve has to be integrated further
      m_ode_solver.solve(v, curve, tau, forward_on_border);
    }
    return curve;
  }
  //============================================================================
  auto vectorfield() const -> const auto& { return m_vectorfield; }
  auto vectorfield()       -> auto&       { return m_vectorfield; }
  //----------------------------------------------------------------------------
  auto ode_solver() const -> const auto& { return *m_ode_solver; }
  auto ode_solver()       ->       auto& { return *m_ode_solver; }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
