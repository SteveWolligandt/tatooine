#include "ensemble_member.h"
#include "ensemble_file_paths.h"
#include "integrate_pathline.h"
#include "positions_in_domain.h"
#include "eddy_props.h"

#include <tatooine/filesystem.h>
#include <mutex>
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
using V = ensemble_member<interpolation::hermite>;
//==============================================================================
std::mutex mutex;
//==============================================================================
auto create_grid() {
  V    v{ensemble_file_paths.front()};
  auto     dim0 = v.xc_axis;
  auto     dim1 = v.yc_axis;
  // linspace dim2{v.z_axis.front(), v.z_axis.back(), 100};
  auto                                                 dim2 = v.z_axis;
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};
  return g;
}
//------------------------------------------------------------------------------
template <typename Xs, typename XIs, typename EulerianQProp, typename LagrangianQProp>
void uncertain_eddy_detection(V const& v,
                              Xs const& xs,
                              XIs const& xis,
                              arithmetic auto const t,
                              arithmetic auto const threshold,
                              EulerianQProp& eulerian_Q_prop,
                              LagrangianQProp& lagrangian_Q_prop) {
  std::atomic_size_t cnt;
  //std::mutex prog_mutex;
  for (size_t i = 0; i < size(xs); ++i) {
    auto const& x = xs[i];
    auto const& xi = xis[i];
    auto const [eulerian_Q, lagrangian_Q, pathline] = eddy_props(v, x, t);
    if (eulerian_Q >= threshold) {
      eulerian_Q_prop.data_at(xi(0), xi(1), xi(2)) += 1;
    }
    if (lagrangian_Q >= threshold) {
      lagrangian_Q_prop.data_at(xi(0), xi(1), xi(2)) += 1;
    }
    ++cnt;
  }
  std::cerr << '\n';
}
//------------------------------------------------------------------------------
void uncertain_eddy_detection(arithmetic auto const threshold) {
  auto  g = create_grid();
  auto& uncertain_eulerian_Q_prop =
      g.add_contiguous_vertex_property<double>("uncertain_eulerian_Q");
  auto& uncertain_lagrangian_Q_prop =
      g.add_contiguous_vertex_property<double>("uncertain_lagrangian_Q");
  g.iterate_over_vertex_indices([&](auto const... is) {
    uncertain_lagrangian_Q_prop.data_at(is...) = 0.0 / 0.0;
    uncertain_eulerian_Q_prop.data_at(is...)   = 0.0 / 0.0;
  }, execution_policy::parallel);

  V           v0{ensemble_file_paths.front()};
  auto const  P   = positions_in_domain(v0, g);
  auto const& xs  = P.first;
  auto const& xis = P.second;
  size_t ti = 0;
  for (auto const t : v0.t_axis) {
    std::cerr << "processing time " << t << ", at index " << ti << " ...\n";
    namespace fs = filesystem;
    fs::path path   = "uncertain_eddy_detection/";
    if (!fs::exists(path)) { fs::create_directory(path); }
    path += "uncertain_eddy_detection_" + std::to_string(ti++) + ".vtk";

    if (!fs::exists(path)) {
#pragma omp parallel for
      for (size_t i = 0; i < size(xis); ++i) {
        auto const& xi                                           = xis[i];
        uncertain_lagrangian_Q_prop.data_at(xi(0), xi(1), xi(2)) = 0;
        uncertain_eulerian_Q_prop.data_at(xi(0), xi(1), xi(2))   = 0;
      }
      for (auto const& ensemble_file_path : ensemble_file_paths) {
        V v{ensemble_file_path};
        std::cerr << "processing file " << ensemble_file_path << '\n';
        uncertain_eddy_detection(v, xs, xis, t, threshold,
                                 uncertain_eulerian_Q_prop,
                                 uncertain_lagrangian_Q_prop);
      }
      for (auto& z : g.dimension<2>()) { z *= -0.0025; }
      g.write_vtk(path);
      for (auto& z : g.dimension<2>()) { z /= -0.0025; }
    }
  }
}
//==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
int main() {
  tatooine::scivis_contest_2020::uncertain_eddy_detection(0.0);
}
