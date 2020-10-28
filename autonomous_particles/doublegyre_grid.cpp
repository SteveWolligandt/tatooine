#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/grid.h>
#include <tatooine/progress_bars.h>
#include <tatooine/vtk_legacy.h>
#include <tatooine/netcdf.h>

#include "ellipsis_vertices.h"
#include "parse_args.h"
//==============================================================================
using namespace tatooine;
//==============================================================================
std::vector<mat34f> initial_ellipses, advected_ellipses,
    back_calculation_ellipses;
//==============================================================================
void create_geometry(auto const& args, auto const& advected_particles) {
  for (auto const& p : advected_particles) {
    auto sqrS =
        inv(p.nabla_phi1()) * p.S() * p.S() * inv(transposed(p.nabla_phi1()));
    auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
    eig_vals = {std::sqrt(eig_vals(0)), std::sqrt(eig_vals(1))};
    transposed(eig_vecs);
    if (args.min_cond > 0 && eig_vals(1) / eig_vals(0) < args.min_cond) {
      continue;
    }

    // advection
    advected_ellipses.push_back(
        mat34f{{p.S()(0, 0), p.S()(0, 1), p.S()(0, 2), p.x1()(0)},
               {p.S()(1, 0), p.S()(1, 1), p.S()(1, 2), p.x1()(1)},
               {p.S()(2, 0), p.S()(2, 1), p.S()(2, 2), p.x1()(2)}});

    // back calculation
    auto Sback = eig_vecs * diag(eig_vals) * transposed(eig_vecs);
    back_calculation_ellipses.push_back(
        mat34f{{Sback(0, 0), Sback(0, 1), Sback(0, 2), p.x0()(0)},
               {Sback(1, 0), Sback(1, 1), Sback(1, 2), p.x0()(1)},
               {Sback(2, 0), Sback(2, 1), Sback(2, 2), p.x0()(2)}});
  }
}
//------------------------------------------------------------------------------
auto write_data() {
  indeterminate_progress_bar([&](auto option) {
    std::vector<size_t> const cnt{1, 4, 3};
    {
      option = "Writing initial circles";
      netcdf::file f_out{"dg_grid_initial.nc", netCDF::NcFile::replace};
      auto         dim_indices = f_out.add_dimension("index");
      auto         dim_cols    = f_out.add_dimension("column", 3);
      auto         dim_rows    = f_out.add_dimension("row", 4);
      auto         var         = f_out.add_variable<float>("transformations",
                                           {dim_indices, dim_rows, dim_cols});

      // create some data
      std::vector<size_t> is{0, 0, 0};
      for (auto const& S : initial_ellipses) {
        var.write(is, cnt, S.data_ptr());
        ++is.front();
      }
    }

    {
      option = "Writing advected ellipses";
      netcdf::file f_out{"dg_grid_advected.nc", netCDF::NcFile::replace};
      auto         dim_indices = f_out.add_dimension("index");
      auto         dim_cols    = f_out.add_dimension("column", 3);
      auto         dim_rows    = f_out.add_dimension("row", 4);
      auto         var         = f_out.add_variable<float>("transformations",
                                           {dim_indices, dim_rows, dim_cols});

      // create some data
      std::vector<size_t> is{0, 0, 0};
      for (auto const& S : initial_ellipses) {
        var.write(is, cnt, S.data_ptr());
        ++is.front();
      }
    }
    {
      option = "Writing back calculated ellipses";
      netcdf::file f_out{"dg_grid_back_calculation.nc",
                         netCDF::NcFile::replace};
      auto         dim_indices = f_out.add_dimension("index");
      auto         dim_cols    = f_out.add_dimension("column", 3);
      auto         dim_rows    = f_out.add_dimension("row", 4);
      auto         var         = f_out.add_variable<float>("transformations",
                                           {dim_indices, dim_rows, dim_cols});

      // create some data
      std::vector<size_t> is{0, 0, 0};
      for (auto const& S : initial_ellipses) {
        var.write(is, cnt, S.data_ptr());
        ++is.front();
      }
    }
  });
}
//------------------------------------------------------------------------------
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

  grid g{linspace{0, 2.0, args.width + 1},
         linspace{0, 1.0, args.height + 1}};
  g.dimension<0>().pop_front();
  g.dimension<1>().pop_front();
  auto const spacing_x = g.dimension<0>().spacing();
  auto const spacing_y = g.dimension<1>().spacing();
  g.dimension<0>().front() -= spacing_x / 2;
  g.dimension<0>().back() -= spacing_x / 2;
  g.dimension<1>().front() -= spacing_y / 2;
  g.dimension<1>().back() -= spacing_y / 2;
  double const r0 = g.dimension<0>().spacing() / 2;

  analytical::fields::numerical::doublegyre v;
  v.set_infinite_domain(true);

  std::atomic_size_t cnt = 0;

  progress_bar([&](auto indicator) {
    for (auto const& x : g.vertices()) {
      autonomous_particle p0{v, x, args.t0, r0};
      p0.phi().use_caching(false);
      initial_ellipses.push_back(
          mat34f{{p0.S()(0, 0), p0.S()(0, 1), p0.S()(0, 2), p0.x1()(0)},
                 {p0.S()(1, 0), p0.S()(1, 1), p0.S()(1, 2), p0.x1()(1)},
                 {p0.S()(2, 0), p0.S()(2, 1), p0.S()(2, 2), p0.x1()(2)}});
      auto const advected_particles = calc_particles(p0);
      create_geometry(args, advected_particles);
      ++cnt;
      indicator.progress = cnt / double(g.num_vertices());
    }
    indicator.progress = 1;
  });
  write_data();
}
