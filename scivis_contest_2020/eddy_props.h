#ifndef TATOOINE_SCIVIS_CONTEST_2020_EDDY_PROPS
#define TATOOINE_SCIVIS_CONTEST_2020_EDDY_PROPS
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/concepts.h>
#include <tatooine/interpolation.h>
#include <tatooine/ode/vclibs/rungekutta43.h>

#include <tuple>
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
template <typename V>
auto eddy_props(V const& v, typename V::pos_t const& x,
                real_number auto const t) {
  auto const Jf             = diff(v, 1e-8);

  auto const eulerian_J     = Jf(x, t);
  auto const eulerian_S     = (eulerian_J + transposed(eulerian_J)) / 2;
  auto const eulerian_Omega = (eulerian_J - transposed(eulerian_J)) / 2;
  auto const eulerian_Q = (sqr_norm(eulerian_Omega) - sqr_norm(eulerian_S)) / 2;

  if (eulerian_Q > 0) {
    parameterized_line<double, 3, interpolation::linear> pathline;

    using solver_t = ode::vclibs::rungekutta43<typename V::real_t, 3>;
    solver_t solver;
    auto     evaluator = [&v, &Jf](auto const& y, auto const t) ->
        typename solver_t::maybe_vec {
          if (!v.in_domain(y, t)) { return ode::vclibs::out_of_domain; }

          auto const J     = Jf(y, t);
          auto const S     = (J + transposed(J)) / 2;
          auto const Omega = (J - transposed(J)) / 2;
          auto const Q     = (sqr_norm(Omega) - sqr_norm(S)) / 2;
          if (Q < 0) { return ode::vclibs::out_of_domain; }

          return v(y, t);
        };

    double const max_ftau = v.t_axis.back() - t;
    double const min_btau = v.t_axis.front() - t;
    double const eps      = 1e-6;
    auto const   ftau     = std::min<double>(24 * 5, max_ftau);

    auto& pathline_Q_prop = pathline.add_vertex_property<double>("Q");
    if (ftau > 0) {
      solver.solve(evaluator, x, t, ftau,
                   [&pathline, &pathline_Q_prop, &Jf, eps](
                       const vec<double, 3>& y, double t) {
                     auto const J     = Jf(y, t);
                     auto const S     = (J + transposed(J)) / 2;
                     auto const Omega = (J - transposed(J)) / 2;
                     auto const Q     = (sqr_norm(Omega) - sqr_norm(S)) / 2;
                     if (Q < 0) {
                       return false;
                     }

                     if (pathline.empty()) {
                       pathline.push_back(y, t, false);
                       pathline_Q_prop.back() = Q;
                       return true;
                     }
                     if (distance(pathline.back_vertex(), y) > eps) {
                       pathline.push_back(y, t, false);
                       pathline_Q_prop.back() = Q;
                       return true;
                     }
                     return false;
                   });
    }
    auto const btau = std::max<double>(-24 * 5, min_btau);
    if (btau < 0) {
      solver.solve(evaluator, x, t, btau,
                   [&pathline, &pathline_Q_prop, &Jf, eps](const vec<double, 3>& y, double t) {
                     auto const J     = Jf(y, t);
                     auto const S     = (J + transposed(J)) / 2;
                     auto const Omega = (J - transposed(J)) / 2;
                     auto const Q     = (sqr_norm(Omega) - sqr_norm(S)) / 2;
                     if (Q < 0) { return false; }
                     if (pathline.empty()) {
                       pathline.push_front(y, t, false);
                       pathline_Q_prop.front() = Q;
                       return true;
                     }
                     if (distance(pathline.front_vertex(), y) > eps) {
                       pathline.push_front(y, t, false);
                       pathline_Q_prop.front() = Q;
                       return true;
                     }
                     return false;
                   });
    }

    auto const t_range = ftau - btau;
    auto       Q_time  = [&](double const threshold) {
      double Q_time = 0;
      for (size_t i = 0; i < pathline.num_vertices() - 1; ++i) {
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
    return std::tuple{eulerian_Q, Q_time(0)};
  }
  return std::tuple{eulerian_Q, 0.0 / 0.0};
}
//==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
#endif
