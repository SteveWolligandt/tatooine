#include <tatooine/analytical/doublegyre.h>
#include <tatooine/analytical/monkey_saddle.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/ridgelines.h>
//==============================================================================
using namespace tatooine;
using analytical::numerical::doublegyre;
using analytical::numerical::monkey_saddle;
//==============================================================================
auto dg_ftle_ridges() {
  auto grid =
      rectilinear_grid{linspace{0.0, 2.0, 2000}, linspace{0.0, 1.0, 1000}};
  auto       dg_flowmap = flowmap(doublegyre{});
  auto const t0  = 0;
  auto const tau = 5;

  auto const fixed_time_phi = [&](auto const& x) {
    return dg_flowmap(x, t0, tau);
  };

  auto const& phi = grid.sample_to_vertex_property(
      fixed_time_phi, "phi", execution_policy::parallel);
  auto const& nabla_phi = grid.sample_to_vertex_property(
      diff(phi), "nabla_phi", execution_policy::parallel);

  auto const ftle_field = [&](integral auto const... is) {
    auto const& nabla_phi_at_pos = nabla_phi(is...);
    auto const  eigvals =
        eigenvalues_sym(transposed(nabla_phi_at_pos) * nabla_phi_at_pos);
    return gcem::log(gcem::sqrt(eigvals(1))) / std::abs(tau);
  };
  auto const& ftle = grid.sample_to_vertex_property(ftle_field, "ftle",
                                                 execution_policy::parallel);
  ridgelines(ftle, execution_policy::parallel).write("dg_ftle_ridges.vtp");
}
//==============================================================================
auto monkey_saddle_ridges() {
  auto grid =
      rectilinear_grid{linspace{-1.0, 1.0, 1000}, linspace{-1.0, 1.0, 1000}};

  auto const& f = grid.sample_to_vertex_property(monkey_saddle{}, "f",
                                                 execution_policy::parallel);
  grid.write("monkey_saddle_ridge_data.vtr");
  ridgelines(f, execution_policy::parallel).write("monkey_saddle_ridges.vtp");
}
//==============================================================================
auto cos_field_ridges() {
  auto grid =
      rectilinear_grid{linspace{-2.0, 2.0, 1000}, linspace{-2.0, 2.0, 1000}};

  auto const& f = grid.sample_to_vertex_property([](auto const& p){return gcem::cos(p.x()) * gcem::cos(p.y());}, "f",
                                                 execution_policy::parallel);
  auto sampler = f.linear_sampler();
  grid.write("cos_field_ridge_data.vtr");
  auto ridges_2d = ridgelines(f, execution_policy::parallel);
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
}
//==============================================================================
auto main() -> int {
  //dg_ftle_ridges();
  //monkey_saddle_ridges();
  cos_field_ridges();
}
