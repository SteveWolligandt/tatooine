#ifndef TATOOINE_SCIVIS_CONTEST_2020_INTEGRATE_PATHLINE_H
#define TATOOINE_SCIVIS_CONTEST_2020_INTEGRATE_PATHLINE_H
#include <tatooine/concepts.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
namespace tatooine::scivis_contest_2020 {
template <typename V>
auto integrate_pathline(V const& v, typename V::pos_t const& x,
                        real_number auto t) {
  parameterized_line<double, 3, interpolation::linear> pathline;

  ode::vclibs::rungekutta43<typename V::real_t, 3> solver;
  double const                            ftau = v.t_axis.back() - t;
  double const                            btau = v.t_axis.front() - t;
  double const                            eps  = 1e-6;
  // std::cerr << "ftau: " << ftau << '\n';
  // std::cerr << "btau: " << btau << '\n';
  // std::cerr << "x: " << x << '\n';
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
}  // namespace tatooine::scivis_contest_2020
#endif
