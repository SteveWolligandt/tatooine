#include <tatooine/analytical/modified_doublegyre.h>
#include <tatooine/analytical/monkey_saddle.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/ridgelines.h>
#include <tatooine/ftle.h>
//==============================================================================
using namespace tatooine;
using analytical::numerical::modified_doublegyre;
using analytical::numerical::monkey_saddle;
//==============================================================================
auto dg_ftle_ridges() {
  auto grid =
      rectilinear_grid{linspace{0.0, 2.0, 500}, linspace{0.0, 1.0, 250}};
  auto const  t0  = 0;
  auto const  tau = 5;
  auto const& f   = ftle(grid, flowmap(modified_doublegyre{}), t0, tau,
                         execution_policy::parallel);
  ridgelines(f, grid, execution_policy::parallel).write("dg_ftle_ridges.vtp");
  grid.write("dg_ftle_ridge_data.vtr");
}
//==============================================================================
auto monkey_saddle_ridges() {
  auto grid =
      rectilinear_grid{linspace{-1.0, 1.0, 1000}, linspace{-1.0, 1.0, 1000}};

  auto const& f = grid.sample_to_vertex_property(monkey_saddle{}, "f",
                                                 execution_policy::parallel);
  ridgelines(f, grid, execution_policy::parallel)
      .write("monkey_saddle_ridges.vtp");
  grid.write("monkey_saddle_ridge_data.vtr");
}
//==============================================================================
auto cos_field_ridges() {
  auto grid = rectilinear_grid{linspace{-1.5 * M_PI, 1.5 * M_PI, 2000},
                               linspace{-1.5 * M_PI, 1.5 * M_PI, 2000}};

  auto const& f = grid.sample_to_vertex_property(
      [](auto const& p) { return gcem::cos(p.x()) * gcem::cos(p.y()); }, "f",
      execution_policy::parallel);
  auto sampler = f.linear_sampler();
  auto ridges_2d = ridgelines(f, grid, execution_policy::parallel);
  auto ridges_3d = edgeset3{};
  for (auto const v : ridges_2d.vertices()) {
    auto const& x = ridges_2d[v];
    ridges_3d.insert_vertex(x.x(), x.y(), sampler(x));
  }
  for (auto const e : ridges_2d.edges()) {
    auto [v0,v1] = ridges_2d[e];
    ridges_3d.insert_edge(edgeset3::vertex_handle{v0.index()},
                          edgeset3::vertex_handle{v1.index()});
  }
  ridges_3d.write("cos_field_ridges.vtp");
  grid.write("cos_field_ridge_data.vtr");
}
//==============================================================================
auto main() -> int {
  dg_ftle_ridges();
  //monkey_saddle_ridges();
  //cos_field_ridges();
}
