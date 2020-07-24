#include <filesystem>

#include "eddy_props.h"
#include "ensemble_file_paths.h"
#include "ensemble_member.h"
#include "integrate_pathline.h"
#include "positions_in_domain.h"
//==============================================================================
using namespace tatooine;
using namespace tatooine::scivis_contest_2020;
using V = ensemble_member<interpolation::hermite>;
//==============================================================================
void print_usage(char** argv);
//------------------------------------------------------------------------------
template <typename V>
auto create_grid(V const& v) {
  auto dim0 = v.xc_axis;
  auto dim1 = v.yc_axis;
  auto dim2 = v.z_axis;
  // linspace dim2{v.z_axis.front(), v.z_axis.back(), 100};
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};
  return g;
}
//------------------------------------------------------------------------------
auto eddy_detection(std::string const& ensemble_id) {
  auto const   ensemble_path = [&] {
    if (std::string{ensemble_id} == "MEAN" ||
        std::string{ensemble_id} == "Mean" ||
        std::string{ensemble_id} == "mean") {
      return tatooine::scivis_contest_2020::mean_file_path;
    }
    return tatooine::scivis_contest_2020::ensemble_file_paths[std::stoi(
        ensemble_id)];
  }();
  V     v{ensemble_path};
  auto  g = create_grid(v);
  auto& eulerian_Q_prop =
      g.template add_contiguous_vertex_property<double, x_fastest>(
          "eulerian_Q");
  auto& lagrangian_Q_prop =
      g.template add_contiguous_vertex_property<double, x_fastest>(
          "lagrangian_Q");
  auto& Q_time_prop_0 =
      g.template add_contiguous_vertex_property<double, x_fastest>("Q_time_0");
  auto& Q_time_prop_5 =
      g.template add_contiguous_vertex_property<double, x_fastest>("Q_time_5");
  auto& Q_time_prop_10 =
      g.template add_contiguous_vertex_property<double, x_fastest>("Q_time_10");

  g.parallel_loop_over_vertex_indices([&](auto const... is) {
    eulerian_Q_prop.data_at(is...)   = 0.0 / 0.0;
    lagrangian_Q_prop.data_at(is...) = 0.0 / 0.0;
    Q_time_prop_0.data_at(is...)     = 0.0 / 0.0;
    Q_time_prop_5.data_at(is...)     = 0.0 / 0.0;
    Q_time_prop_10.data_at(is...)    = 0.0 / 0.0;
  });

  auto const  P   = positions_in_domain(v, g);
  auto const& xs  = P.first;
  auto const& xis = P.second;
  for (auto& z : g.dimension<2>()) { z *= -0.0025; }

  size_t ti = 0;
  for (auto t : v.t_axis) {
    std::cerr << "processing time " << t << ", at index " << ti << " ...\n";
    namespace fs = std::filesystem;
    fs::path p   = "eddy_detection_" + std::string{ensemble_id} + "/";
    if (!fs::exists(p)) { fs::create_directory(p); }
    p += "eddy_detection_" + std::to_string(ti++) + ".vtk";

    if (!fs::exists(p)) {
      std::atomic_size_t cnt  = 0;
      bool               done = false;
      std::thread        monitor{[&] {
        while (!done) {
          std::cerr << static_cast<double>(cnt) / size(xs) * 100
                    << "  %        \r";
          std::this_thread::sleep_for(std::chrono::milliseconds{200});
        }
      }};
#pragma omp parallel for
      for (size_t i = 0; i < size(xs); ++i) {
        auto const& x                                   = xs[i];
        auto const& xi                                  = xis[i];
        auto const [eulerian_Q, lagrangian_Q, Q_time_0, Q_time_5, Q_time_10,
                    pathline]                           = eddy_props(v, x, t);

        auto const t_range = (pathline.back_parameterization() -
                              pathline.front_parameterization());
        eulerian_Q_prop.data_at(xi(0), xi(1), xi(2))   = eulerian_Q;
        lagrangian_Q_prop.data_at(xi(0), xi(1), xi(2)) = lagrangian_Q;
        Q_time_prop_0.data_at(xi(0), xi(1), xi(2))     = Q_time_0 / t_range;
        Q_time_prop_5.data_at(xi(0), xi(1), xi(2))     = Q_time_5 / t_range;
        Q_time_prop_10.data_at(xi(0), xi(1), xi(2))    = Q_time_10 / t_range;
        ++cnt;
      }
      done = true;
      monitor.join();

      g.write_vtk(p);
    }
  }
}
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // check arguments
  if (argc < 2) {
    print_usage(argv);
    throw std::runtime_error{"specify path to ensemble member!"};
  }
  eddy_detection(argv[1]);
}
//------------------------------------------------------------------------------
void print_usage(char** argv) {
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " <path/to/ensemble>\n";
}
