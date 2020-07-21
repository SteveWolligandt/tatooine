#include "ensemble_member.h"

#include <filesystem>
#include <mutex>

#include "ensemble_file_paths.h"
//==============================================================================
namespace fs = std::filesystem;
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
using V = ensemble_member<interpolation::hermite>;
//==============================================================================
std::mutex pathline_mutex;
std::mutex prog_mutex;
//==============================================================================
auto create_grid() {
  V    v{ensemble_file_paths.front()};

  auto dim0 = v.xc_axis;
  auto dim1 = v.yc_axis;
  auto dim2 = linspace{v.z_axis.front(), v.z_axis.back(), 100};
  dim0.pop_front();
  dim0.pop_back();
  dim0.pop_back();

  dim1.pop_front();
  dim1.front() = 14;
  dim1.pop_back();
  dim1.pop_back();

  dim2.pop_back();
  grid g{dim0, dim1, dim2};
  return g;
}
//------------------------------------------------------------------------------
auto integrate_pathline(V const& v, typename V::pos_t const& x,
                        real_number auto t) {
  parameterized_line<double, 3, interpolation::linear> pathline;

  double const ftau = std::min<double>(50, v.t_axis.back() - t);
  double const btau = std::max<double>(-50, v.t_axis.front() - t);
  ode::vclibs::rungekutta43<V::real_t, 3> solver;
  double const eps = 1e-6;
  //std::cerr << "ftau: " << ftau << '\n';
  //std::cerr << "btau: " << btau << '\n';
  //std::cerr << "x: " << x << '\n';
  solver.solve(
      v, x, t, ftau, [&pathline, eps](auto t, const auto& y) {
        if (pathline.empty() || distance(pathline.back_vertex(), y) > eps) {
          pathline.push_back(y, t);
          return true;
        }
        return false;
      });
  solver.solve(
      v, x, t, btau, [&pathline, eps](auto t, const auto& y) {
        if (pathline.empty() || distance(pathline.front_vertex(), y) > eps) {
          pathline.push_front(y, t);
          return true;
        }
        return false;
      });
  return pathline;
}
//------------------------------------------------------------------------------
template <typename Prop, typename... Is>
void collect_pathlines_in_eddy(
    V const& v, typename V::pos_t const& x, real_number auto const t,
    real_number auto const threshold,
    std::vector<parameterized_line<double, 3, interpolation::linear>>&
          pathlines,
    Prop& prop, Is... is) {
  auto const Jf = diff(v, 1e-8);

  auto  pathline      = integrate_pathline(v, x, t);
  auto& pathline_prop = pathline.template add_vertex_property<double>("prop");

  // for each vertex of the pathline calculate properties
  for (size_t i = 0; i < pathline.num_vertices(); ++i) {
    typename std::decay_t<decltype(pathline)>::vertex_idx vi{i};
    auto const& px = pathline.vertex_at(i);
    auto const& pt = pathline.parameterization_at(i);
    if (v.in_domain(px, pt)) {
      auto const J = Jf(px, pt);

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

    prop.data_at(is...) = lagrangian_eddy;
    if (lagrangian_eddy > threshold) {
      std::lock_guard lock{pathline_mutex};
      pathlines.push_back(std::move(pathline));
    }
  }
}
//------------------------------------------------------------------------------
template <typename Grid, typename Prop>
void collect_pathlines_in_eddy(
    V const& v, real_number auto const t, real_number auto const threshold,
    Grid const& g,
    std::vector<parameterized_line<double, 3, interpolation::linear>>&
          pathlines,
    Prop& prop) {
  size_t             cnt       = 0;
  auto               iteration = [&](auto const... is) {
    auto const x = g.vertex_at(is...);
    if (v.in_domain(x, t)) {
      collect_pathlines_in_eddy(v, x, t, threshold, pathlines, prop, is...);
    }
    {
      std::lock_guard lock{prog_mutex};
      ++cnt;
      std::cerr << (100 * cnt / static_cast<double>(g.num_vertices()))
                << " %       \r";
    }
  };

  parallel_for_loop(iteration, g.template size<0>(), g.template size<1>(),
                    g.template size<2>());
  std::cerr << '\n';
}
//------------------------------------------------------------------------------
void collect_pathlines_in_eddy(std::string const&     filepath,
                               real_number auto const threshold) {
  fs::path p = filepath;
  std::vector<parameterized_line<double, 3, interpolation::linear>> pathlines;
  auto  g = create_grid();
  auto& prop =
      g.add_contiguous_vertex_property<double, x_fastest>("lagrangian_Q");

  V          v{p};
  auto const t = (v.t_axis.back() + v.t_axis.front()) / 2;
  collect_pathlines_in_eddy(v, t, threshold, g, pathlines, prop);
  std::string outpath_pathlines = fs::path{p.filename()}.replace_extension(
      "pathlines_in_eddy_" + std::to_string(t) + ".vtk");
  write_vtk(pathlines, outpath_pathlines);

  std::string outpath_scalar = fs::path{p.filename()}.replace_extension(
      "pathlines_in_eddy_scalar" + std::to_string(t) + ".vtk");
  g.write_vtk(outpath_scalar);
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
    if (std::string{argv[1]} == "MEAN" || std::string{argv[1]} == "mean" ||
        std::string{argv[1]} == "Mean") {
      return tatooine::scivis_contest_2020::mean_file_path;
    }
    return tatooine::scivis_contest_2020::ensemble_file_paths[std::stoi(
        argv[1])];
  }();
  tatooine::scivis_contest_2020::collect_pathlines_in_eddy(ensemble_path, 0.05);
}
