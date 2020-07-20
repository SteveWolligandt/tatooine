#include <tatooine/fields/scivis_contest_2020_ensemble_member.h>
#include <filesystem>
#include <mutex>
#include "ensemble_file_paths.h"
//==============================================================================
namespace fs = std::filesystem;
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
using V = tatooine::fields::scivis_contest_2020_ensemble_member<
    interpolation::hermite>;
//==============================================================================
std::mutex pathline_mutex;
std::mutex prog_mutex;
//==============================================================================
auto create_grid() {
  V    v{ensemble_file_paths.front()};
  grid g{v.xc_axis, v.yc_axis,
         linspace{v.z_axis.front(), v.z_axis.back(), 100}};

  g.dimension<0>().pop_front();
  g.dimension<0>().pop_back();
  g.dimension<0>().pop_back();

  g.dimension<1>().pop_front();
  g.dimension<1>().pop_back();
  g.dimension<1>().pop_back();

  g.dimension<2>().pop_front();
  g.dimension<2>().pop_back();
  g.dimension<2>().pop_back();
  return g;
}
//------------------------------------------------------------------------------
auto integrate_pathline(V const& v, typename V::pos_t const& x,
                        real_number auto t) {
  parameterized_line<double, 3, interpolation::linear> pathline;

  ode::vclibs::rungekutta43<V::real_t, 3> solver;
  solver.solve(v, vec{x(0), x(1), x(2)}, t, 10,
               [&pathline](auto t, const auto& y) {
                 if (pathline.empty()) {
                   pathline.push_back(y, t);
                 } else if (distance(pathline.back_vertex(), y) > 1e-6) {
                   pathline.push_back(y, t);
                 }
               });
  solver.solve(v, vec{x(0), x(1), x(2)}, t, -10,
               [&pathline](auto t, const auto& y) {
                 if (pathline.empty()) {
                   pathline.push_back(y, t);
                 } else if (distance(pathline.front_vertex(), y) > 1e-6) {
                   pathline.push_front(y, t);
                 }
               });
  return pathline;
}
//------------------------------------------------------------------------------
void collect_pathlines_in_eddy(
    V const& v, typename V::pos_t const& x, real_number auto const t,
    real_number auto const threshold,
    std::vector<parameterized_line<double, 3, interpolation::linear>>&
        pathlines) {
  auto const Jf = diff(v, 1e-8);

  auto pathline      = integrate_pathline(v, x, t);
  auto& pathline_prop = pathline.template add_vertex_property<double>("prop");

  // for each vertex of the pathline calculate properties
  for (size_t i = 0; i < pathline.num_vertices(); ++i) {
    typename std::decay_t<decltype(pathline)>::vertex_idx vi{i};
    auto const& px = pathline.vertex_at(i);
    auto const& pt = pathline.parameterization_at(i);
    if (v.in_domain(px, pt)) {
      auto const J     = Jf(px, pt);

      auto const S     = (J + transposed(J)) / 2;
      auto const Omega = (J - transposed(J)) / 2;
      // auto const SS    = S * S;
      // auto const OO    = Omega * Omega;
      // auto const SSOO  = SS + OO;

      // vec const vort{J2, 1) - J(1, 2),
      //               J(0, 2) - J(2, 0),
      //               J(1, 0) - J(0, 1)};
      // vorticity
      // pathline_prop[vi] = length(vort);
      // Q
      pathline_prop[vi] = (sqr_norm(Omega) - sqr_norm(S)) / 2;
      // lambda2
      // pathline_prop[vi] = eigenvalues_sym(SSOO)(1);
    } else {
      pathline_prop[vi] = 0.0 / 0.0;
    }
  }

  if (pathline.num_vertices() > 1) {
    auto const lagrangian_eddy = pathline.integrate_property(pathline_prop);
    if (lagrangian_eddy > threshold) {
      std::lock_guard lock{pathline_mutex};
      pathlines.push_back(std::move(pathline));
    }
  }
}
//------------------------------------------------------------------------------
template <typename Grid>
void collect_pathlines_in_eddy(
    V const& v, real_number auto const t, real_number auto const threshold,
    Grid const& g,
    std::vector<parameterized_line<double, 3, interpolation::linear>>&
        pathlines) {
  std::atomic_size_t cnt;
  auto               iteration = [&](auto const... is) {
    auto const x = g.vertex_at(is...);
    if (v.in_domain(x, t)) {
      collect_pathlines_in_eddy(v, x, t, threshold, pathlines);
    }
    ++cnt;
    std::lock_guard lock{prog_mutex};
    std::cerr << std::setprecision(4)
              << (100 * cnt / static_cast<double>(g.num_vertices()))
              << " %    \r";
  };
  parallel_for_loop(iteration, g.template size<0>(), g.template size<1>(),
                    g.template size<2>());
  std::cerr << '\n';
}
//------------------------------------------------------------------------------
void collect_pathlines_in_eddy(std::string const& filepath,
                               real_number auto const threshold) {
  fs::path    p       = filepath;
  std::cerr << p << '\n';
  std::vector<parameterized_line<double, 3, interpolation::linear>> pathlines;
  auto g = create_grid();

  V          v{p};
  auto const t = (v.t_axis.back() + v.t_axis.front()) / 2;
  collect_pathlines_in_eddy(v, t, threshold, g, pathlines);
  std::string outpath = fs::path{p.filename()}.replace_extension(
      "pathlines_in_eddy_" + std::to_string(t) + ".vtk");
  write_vtk(pathlines, outpath);
}
//==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
int main(int argc, char const** argv) {
  if (argc < 2) {
    throw std::runtime_error{
        "you need to specify ensemble member number or MEAN"};
  }
  if (argc < 3) { throw std::runtime_error{"you need to specify time"}; }

  auto const ensemble_path = [&] {
    if (std::string{argv[1]} == "MEAN" ||
        std::string{argv[1]} == "mean" ||
        std::string{argv[1]} == "Mean") {
      return tatooine::scivis_contest_2020::mean_file_path;
    }
    return tatooine::scivis_contest_2020::ensemble_file_paths[std::stoi(
        argv[1])];
  }();
  tatooine::scivis_contest_2020::collect_pathlines_in_eddy(ensemble_path, std::stoi(argv[2]));
}
