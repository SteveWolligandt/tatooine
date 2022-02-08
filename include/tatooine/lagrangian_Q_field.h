#ifndef TATOOINE_LAGRANGIAN_Q_FIELD_H
#define TATOOINE_LAGRANGIAN_Q_FIELD_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
#include <tatooine/line.h>
#include <tatooine/interpolation.h>
#include <tatooine/Q_field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, size_t N>
class lagrangian_Q_field
    : public scalarfield<lagrangian_Q_field<V, N>, typename V::real_type, N> {
  //============================================================================
  // typedefs
  //============================================================================
 public:
  using real_type = typename V::real_type;
  using this_type = lagrangian_Q_field<V, N>;
  using parent_type =
      field<this_type, real_type, V::num_dimensions()>;
  using parent_type::num_dimensions;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  using ode_solver_t = ode::vclibs::rungekutta43<real_type, 3>;
  //============================================================================
  // fields
  //============================================================================
 private:
  V const& m_v;
  real_type m_btau, m_ftau;

  //============================================================================
  // ctor
  //============================================================================
 public:
  template <typename Real>
  lagrangian_Q_field(const vectorfield<V, Real, N>& v,
                     arithmetic auto const btau, arithmetic auto const ftau)
      : m_v{v.as_derived()},
        m_btau{static_cast<real_type>(btau)},
        m_ftau{static_cast<real_type>(ftau)} {}

  //============================================================================
  // methods
  //============================================================================
 public:
  constexpr tensor_type evaluate(const pos_type& x, real_type t) const {
    parameterized_line<real_type, num_dimensions(), interpolation::linear> pathline;
    auto const Qf = Q(m_v);
    ode_solver_t ode_solver;
    auto     evaluator = [this, &Qf](auto const& y, auto const t) ->
        typename ode_solver_t::maybe_vec {
          if (!m_v.in_domain(y, t)) { return ode::vclibs::out_of_domain; }
          return m_v(y, t);
        };

    real_type const eps      = 1e-6;
    auto& pathline_Q_prop = pathline.template add_vertex_property<real_type>("Q");
    if (m_ftau > 0) {
      ode_solver.solve(evaluator, x, t, m_ftau,
                   [&pathline, &pathline_Q_prop, &Qf, eps](
                       const pos_type& y, real_type t) {
                     auto const Q     = Qf(y, t);

                     if (pathline.empty()) {
                       pathline.push_back(y, t, false);
                       pathline_Q_prop.back() = Q;
                     }
                     if (distance(pathline.back_vertex(), y) > eps) {
                       pathline.push_back(y, t, false);
                       pathline_Q_prop.back() = Q;
                     }
                   });
    }
    if (m_btau < 0) {
      ode_solver.solve(evaluator, x, t, m_btau,
                   [&pathline, &pathline_Q_prop, &Qf, eps](const vec<double, 3>& y, double t) {
                     auto const Q     = Qf(y, t);
                     if (pathline.empty()) {
                       pathline.push_front(y, t, false);
                       pathline_Q_prop.front() = Q;
                     }
                     if (distance(pathline.front_vertex(), y) > eps) {
                       pathline.push_front(y, t, false);
                       pathline_Q_prop.front() = Q;
                     }
                   });
    }

    auto const t_range = m_ftau - m_btau;
    auto       Q_time  = [&](double const threshold) {
      double Q_time = 0;
      for (size_t i = 0; i < pathline.vertices().size() - 1; ++i) {
        typename decltype(pathline)::vertex_idx vi{i};
        typename decltype(pathline)::vertex_idx vj{i + 1};
        auto const& t0 = pathline.parameterization_at(i);
        auto const& t1 = pathline.parameterization_at(i + 1);
        auto const& Q0 = pathline_Q_prop[vi];
        auto const& Q1 = pathline_Q_prop[vj];
        if (Q0 >= threshold && Q1 >= threshold) {
          Q_time += t1 - t0;
        } else if (Q0 >= threshold && Q1 < threshold) {
          auto const t_root =
              ((t1 - t0) * threshold - Q0 * t1 + Q1 * t0) / (Q1 - Q0);
          Q_time += t_root - t0;
        } else if (Q0 < threshold && Q1 >= threshold) {
          auto const t_root =
              ((t1 - t0) * threshold - Q0 * t1 + Q1 * t0) / (Q1 - Q0);
          Q_time += t1 - t_root;
        }
      }
      return Q_time / t_range;
    };
    return Q_time(0);
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_type& x, real_type t) const {
    const real_type eps = 1e-6;
    return m_v.in_domain(x + vec{eps, 0, 0}, t) &&
           m_v.in_domain(x - vec{eps, 0, 0}, t) &&
           m_v.in_domain(x + vec{0, eps, 0}, t) &&
           m_v.in_domain(x - vec{0, eps, 0}, t) &&
           m_v.in_domain(x + vec{0, 0, eps}, t) &&
           m_v.in_domain(x - vec{0, 0, eps}, t);
  }
};
//==============================================================================
template <typename V, typename Real, size_t N>
auto lagrangian_Q(const field<V, Real, N, N>& vf, arithmetic auto const btau,
                  arithmetic auto const ftau) {
  return lagrangian_Q_field<V, N>{vf, btau, ftau};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
