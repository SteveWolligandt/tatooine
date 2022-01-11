#include <tatooine/filesystem.h>

#include <mutex>

#include "ensemble_file_paths.h"
#include "ensemble_member.h"
#include "integrate_pathline.h"
#include "positions_in_domain.h"
//==============================================================================
namespace fs = filesystem;
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
using V = ensemble_member<interpolation::hermite>;
//==============================================================================
std::mutex pathline_mutex;
std::mutex prog_mutex;
//==============================================================================
auto create_grid() {
  V v{ensemble_file_paths.front()};

  auto dim0 = v.xc_axis;
  dim0.size() /= 4;
  auto dim1 = v.yc_axis;
  dim1.size() /= 4;
  linspace dim2{v.z_axis.front(), v.z_axis.back(), 100};

  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};
  g.add_contiguous_vertex_property<double>("lagrangian_Q");
  return g;
}
//------------------------------------------------------------------------------
template <typename... Is>
void collect_pathlines_in_eddy(
    V const& v, typename V::pos_t const& x, arithmetic auto const t,
    arithmetic auto const threshold,
    std::vector<parameterized_line<double, 3, interpolation::linear>>&
        pathlines) {
  auto const Jf = diff(v, 1e-8);

  auto  pathline      = integrate_pathline(v, x, t);
  auto& pathline_prop = pathline.template add_vertex_property<double>("prop");

  // for each vertex of the pathline calculate properties
  for (size_t i = 0; i < pathline.vertices().size(); ++i) {
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

  if (pathline.vertices().size() > 1) {
    auto const lagrangian_eddy = pathline.integrate_property(pathline_prop);

    if (lagrangian_eddy > threshold) {
      std::lock_guard lock{pathline_mutex};
      pathlines.push_back(std::move(pathline));
    }
  }
}
//------------------------------------------------------------------------------
void collect_pathlines_in_eddy(
    V const& v, arithmetic auto const t, arithmetic auto const threshold,
    std::vector<vec<double, 3>> const& xs,
    std::vector<parameterized_line<double, 3, interpolation::linear>>&
        pathlines) {
  std::atomic_size_t cnt  = 0;
  bool               done = false;

  std::thread monitor{[&] {
    while (!done) {
      std::cerr << "  integrating pathlines... "
                << (100 * cnt / static_cast<double>(size(xs))) << " %       \r";
      std::this_thread::sleep_for(std::chrono::milliseconds{200});
    }
  }};

#pragma omp parallel for schedule(dynamic, 2)
  for (size_t i = 0; i < size(xs); ++i) {
    collect_pathlines_in_eddy(v, xs[i], t, threshold, pathlines);
    ++cnt;
  }
  done = true;
  monitor.join();
  std::cerr << '\n';
}
//------------------------------------------------------------------------------
void collect_pathlines_in_eddy(char const*            ensemble_id,
                               arithmetic auto const threshold) {
  auto const ensemble_path = [&] {
    if (std::string{ensemble_id} == "MEAN" ||
        std::string{ensemble_id} == "Mean" ||
        std::string{ensemble_id} == "mean") {
      return tatooine::scivis_contest_2020::mean_file_path;
    }
    return tatooine::scivis_contest_2020::ensemble_file_paths[std::stoi(
        ensemble_id)];
  }();
  fs::path const dir_path =
      std::string{"pathlines_in_eddy_"} + std::string{ensemble_id};
  if (!fs::exists(dir_path)) { fs::create_directory(dir_path); }
  std::vector<parameterized_line<double, 3, interpolation::linear>> pathlines;
  auto g = create_grid();
  V    v{ensemble_path};

  size_t ti = 0;
  for (auto const t : v.t_axis) {
    std::cerr << "processing time " << t << ", at index " << ti << " ...\n";
    fs::path outpath_pathlines = dir_path;
    outpath_pathlines /= "pathlines_in_eddy_" + std::to_string(ti++) + ".vtk";
    if (!fs::exists(outpath_pathlines)) {
      collect_pathlines_in_eddy(v, t, threshold,
                                positions_in_domain(v, g).first, pathlines);

      write_vtk(pathlines, outpath_pathlines.c_str());
    }
  }
}
//==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
int main(int argc, char const** argv) {
  if (argc < 2) {
    throw std::runtime_error{
        "You need to specify either mean or ensemble index."};
  }
  if (argc < 3) { throw std::runtime_error{"you need to specify a threshold"}; }
  tatooine::scivis_contest_2020::collect_pathlines_in_eddy(argv[1],
                                                           std::stoi(argv[2]));
}
