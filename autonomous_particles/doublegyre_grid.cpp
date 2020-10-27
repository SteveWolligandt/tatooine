#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/grid.h>
#include <tatooine/progress_bars.h>
#include <tatooine/vtk_legacy.h>

#include "ellipsis_vertices.h"
#include "parse_args.h"
//==============================================================================
using namespace tatooine;
//==============================================================================
size_t const                ellipsis_res = 12;
std::vector<vec<double, 3>> initial_ellipses, advected_ellipses,
    back_calculation_ellipses;
//==============================================================================
void create_geometry(auto const& args, auto const& advected_particles) {
  for (auto const& p : advected_particles) {
    {
      auto sqrS =
          inv(p.nabla_phi1()) * p.S() * p.S() * inv(transposed(p.nabla_phi1()));
      auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
      eig_vals = {std::sqrt(eig_vals(0)), std::sqrt(eig_vals(1))};
      transposed(eig_vecs);
      if (args.min_cond > 0 && eig_vals(1) / eig_vals(0) < args.min_cond) {
        continue;
      }
    }

    {
      auto vs =
          ellipsis_vertices(p.S(), p.x1(), ellipsis_res);
      std::move(begin(vs), end(vs), std::back_inserter(advected_ellipses));
    }

    {
      auto sqrS =
          inv(p.nabla_phi1()) * p.S() * p.S() * inv(transposed(p.nabla_phi1()));
      auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
      eig_vals      = {std::sqrt(eig_vals(0)), std::sqrt(eig_vals(1))};
      auto S        = eig_vecs * diag(eig_vals) * transposed(eig_vecs);
      auto vs = ellipsis_vertices(S, p.x0(), ellipsis_res);
      std::move(begin(vs), end(vs),
                std::back_inserter(back_calculation_ellipses));
    }
  }
}
//------------------------------------------------------------------------------
auto write_data() {
  indeterminate_progress_bar([&](auto option) {
    {
      vtk::legacy_file_writer initial_file{"dg_grid_initial.vtk",
                                           vtk::POLYDATA};
      initial_file.write_header();
      option = "Writing initial points";
      initial_file.write_points(initial_ellipses);
      option = "Writing initial lines";
      std::vector<std::vector<size_t>> indices_initial;
      size_t k = 0;
      for (size_t i = 0; i < size(initial_ellipses) / ellipsis_res; ++i) {
          indices_initial.emplace_back();
        for (size_t j = 0; j < ellipsis_res; ++j) {
          indices_initial.back().push_back(k++);
        }
        indices_initial.back().push_back(k-ellipsis_res);
      }
      initial_file.write_lines(indices_initial);
    }

    std::vector<std::vector<size_t>> indices;
    size_t                           k = 0;
    for (size_t i = 0; i < size(advected_ellipses) / ellipsis_res; ++i) {
      indices.emplace_back();
      for (size_t j = 0; j < ellipsis_res; ++j) {
        indices.back().push_back(k++);
      }
      indices.back().push_back(k - ellipsis_res);
    }
    {
      vtk::legacy_file_writer advection_file{"dg_grid_advected.vtk",
                                             vtk::POLYDATA};
      advection_file.write_header();
      option = "Writing advected points";
      advection_file.write_points(advected_ellipses);
      option = "Writing advected lines";
      advection_file.write_lines(indices);
      advection_file.close();
    }
    {
      vtk::legacy_file_writer back_calc_file{"dg_grid_back_calculation.vtk",
                                             vtk::POLYDATA};
      back_calc_file.write_header();
      option = "Writing back calculated points";
      back_calc_file.write_points(back_calculation_ellipses);
      option = "Writing back calculated lines";
      back_calc_file.write_lines(indices);
      back_calc_file.close();
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
      auto vs = ellipsis_vertices(p0.S(), p0.x1(), ellipsis_res);
      std::move(begin(vs), end(vs), std::back_inserter(initial_ellipses));

      auto const advected_particles = calc_particles(p0);
      //create_geometry(args, advected_particles);
      ++cnt;
      indicator.progress = cnt / double(g.num_vertices());
    }
    indicator.progress = 1;
  });
  //write_data();
}
