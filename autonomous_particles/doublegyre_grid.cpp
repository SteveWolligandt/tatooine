#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/grid.h>
#include <tatooine/netcdf.h>
#include <tatooine/progress_bars.h>
#include <tatooine/vtk_legacy.h>

#include "ellipsis_vertices.h"
#include "parse_args.h"
//==============================================================================
std::vector<size_t> const cnt{1, 2, 3};
std::vector<std::thread>  writers;
std::mutex                writer_mutex;
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto args_opt = parse_args(argc, argv);
  if (!args_opt) {
    return 1;
  }
  auto args = *args_opt;
  auto calc_particles =
      [&args](auto const& p0) -> std::vector<std::decay_t<decltype(p0)>> {
    switch (args.num_splits) {
      case 2:
        return p0.advect_with_2_splits(args.tau_step, args.t0 + args.tau);
      case 3:
        return p0.advect_with_3_splits(args.tau_step, args.t0 + args.tau);
      case 5:
        return p0.advect_with_5_splits(args.tau_step, args.t0 + args.tau);
      case 7:
        return p0.advect_with_7_splits(args.tau_step, args.t0 + args.tau);
    }
    return {};
  };

  std::vector<size_t> initial_netcdf_is{0, 0, 0}, advected_netcdf_is{0, 0, 0},
      back_calculation_netcdf_is{0, 0, 0};
  netcdf::file initial_ellipsis_file{"dg_grid_initial.nc",
                                     netCDF::NcFile::replace},
      advected_file{"dg_grid_advected.nc", netCDF::NcFile::replace},
      back_calculation_file{"dg_grid_back_calculation.nc",
                            netCDF::NcFile::replace};
  auto initial_var = initial_ellipsis_file.add_variable<float>(
      "transformations", {initial_ellipsis_file.add_dimension("index"),
                          initial_ellipsis_file.add_dimension("row", 2),
                          initial_ellipsis_file.add_dimension("column", 3)});

  auto advected_var = advected_file.add_variable<float>(
      "transformations", {advected_file.add_dimension("index"),
                          advected_file.add_dimension("row", 2),
                          advected_file.add_dimension("column", 3)});
  auto back_calculation_var = back_calculation_file.add_variable<float>(
      "transformations", {back_calculation_file.add_dimension("index"),
                          back_calculation_file.add_dimension("row", 2),
                          back_calculation_file.add_dimension("column", 3)});

  grid g{linspace{0, 2.0, args.width + 1}, linspace{0, 1.0, args.height + 1}};
  g.dimension<0>().pop_front();
  g.dimension<1>().pop_front();
  auto const spacing_x = g.dimension<0>().spacing();
  auto const spacing_y = g.dimension<1>().spacing();
  g.dimension<0>().front() -= spacing_x / 2;
  g.dimension<0>().back() -= spacing_x / 2;
  g.dimension<1>().front() -= spacing_y / 2;
  g.dimension<1>().back() -= spacing_y / 2;
  double const               r0 = g.dimension<0>().spacing() / 2;
  triangular_mesh<double, 2> mesh;
  auto& flowmap = mesh.add_vertex_property<vec<double, 2>>("flowmap");

  analytical::fields::numerical::doublegyre v;
  v.set_infinite_domain(true);

  std::atomic_size_t particle_counter = 0;

  progress_bar([&](auto indicator) {
    for (auto const& x : g.vertices()) {
      autonomous_particle p0{v, x, args.t0, r0};

      mat23f T{{p0.S()(0, 0), p0.S()(0, 1), p0.x1()(0)},
               {p0.S()(1, 0), p0.S()(1, 1), p0.x1()(1)}};
      initial_var.write(initial_netcdf_is, initial_netcdf_is, T.data_ptr());
      ++initial_netcdf_is.front();

      p0.phi().use_caching(false);

      auto ps = calc_particles(p0);
      mesh.points().reserve(mesh.num_vertices() + size(ps));
      for (auto const& p : ps) {
        auto v     = mesh.insert_vertex(p.x0());
        flowmap[v] = p.x1();
      }

      for (auto const& p : ps) {
        auto sqrS = inv(p.nabla_phi1()) * p.S() * p.S() *
                    inv(transposed(p.nabla_phi1()));
        auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
        eig_vals = {std::sqrt(eig_vals(0)), std::sqrt(eig_vals(1))};
        transposed(eig_vecs);
        if (args.min_cond > 0 && eig_vals(1) / eig_vals(0) < args.min_cond) {
          continue;
        }

        // advection
        mat23f advected_T{{p.S()(0, 0), p.S()(0, 1), p.x1()(0)},
                          {p.S()(1, 0), p.S()(1, 1), p.x1()(1)}};
        {
          std::lock_guard lock{writer_mutex};
          advected_var.write(advected_netcdf_is, cnt, advected_T.data_ptr());
          ++advected_netcdf_is.front();
        }

        // back calculation
        auto   Sback = eig_vecs * diag(eig_vals) * transposed(eig_vecs);
        mat23f back_calculation_T{{Sback(0, 0), Sback(0, 1), p.x1()(0)},
                                  {Sback(1, 0), Sback(1, 1), p.x1()(1)}};
        {
          std::lock_guard lock{writer_mutex};
          back_calculation_var.write(back_calculation_netcdf_is, cnt,
                                     back_calculation_T.data_ptr());
          ++back_calculation_netcdf_is.front();
        }
      }
      ps.clear();
      ps.shrink_to_fit();

      ++particle_counter;
      indicator.progress = particle_counter / double(g.num_vertices());
    }
    indicator.progress = 1;
  });
  mesh.triangulate_delaunay();
  mesh.write_vtk("doublegyre_flowmap.vtk");
  for (auto& writer : writers) {
    writer.join();
  }
}
