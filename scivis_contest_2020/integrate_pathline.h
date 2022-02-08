#ifndef TATOOINE_SCIVIS_CONTEST_2020_INTEGRATE_PATHLINE_H
#define TATOOINE_SCIVIS_CONTEST_2020_INTEGRATE_PATHLINE_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
template <typename V>
auto integrate_pathline(V&& v, typename V::pos_type const& x, arithmetic auto t,
                        double btau, double ftau) {
  parameterized_line<double, 3, interpolation::linear> pathline;

  ode::vclibs::rungekutta43<typename V::real_type, 3> solver;
  double const                            max_ftau = v.t_axis.back() - t;
  double const                            min_btau = v.t_axis.front() - t;
  double const                            eps  = 1e-6;
  ftau = std::min(ftau, max_ftau);
  btau = std::max(btau, min_btau);

  if (ftau > 0) {
    solver.solve(v, x, t, ftau, [&pathline, eps](auto t, const auto& y) {
      if (pathline.empty() || distance(pathline.back_vertex(), y) > eps) {
        pathline.push_back(y, t);
        return true;
      }
      return false;
    });
  }
  if (btau < 0) {
    solver.solve(v, x, t, btau, [&pathline, eps](auto t, const auto& y) {
      if (pathline.empty() || distance(pathline.front_vertex(), y) > eps) {
        pathline.push_front(y, t);
        return true;
      }
      return false;
    });
  }
  return pathline;
}
//==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
#endif
