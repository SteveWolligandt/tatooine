#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
//==============================================================================
using namespace tatooine;
using tatooine::analytical::fields::numerical::doublegyre;
//==============================================================================
auto measure(auto const& sampler, auto const& phi, auto const t0, auto const t1,
             auto& grid, auto& local_errors) {
  auto const ellipse = sampler.ellipse1();
  grid.vertices().iterate_indices(
      [&](auto const... is) {
        auto phi2 = phi;
        phi2.use_caching(false);
        auto       lambda     = eigenvalues(ellipse.S());
        auto const local_x    = grid.vertex_at(is...);
        auto const physical_x = ellipse.S() * local_x + ellipse.center();
        local_errors(is...) +=
            euclidean_distance(phi2(physical_x, t1, t0 - t1),
                               sampler.sample_backward(physical_x)) /
            (lambda(0).real() * lambda(1).real());
      },
      execution_policy::parallel);
}
//------------------------------------------------------------------------------
auto main(int const /*argc*/, char const** argv) -> int {
  auto v        = doublegyre{};
  auto phi      = flowmap(v);
  auto tau_step = real_t(0.1);
  auto t0       = std::stod(argv[1]);
  auto t1       = std::stod(argv[2]);
  auto x0       = vec2{std::stod(argv[3]), std::stod(argv[4])};
  auto r0       = std::stod(argv[5]);
  auto local_neighborhood =
      rectilinear_grid{linspace{-3.0, 3.0, 1000}, linspace{-3.0, 3.0, 1000}};
  auto& local_errors =
      local_neighborhood.scalar_vertex_property("local_errors");

  auto part           = autonomous_particle<real_t, 2>{x0, t0, r0};
  auto advected_parts = part.advect_with_3_splits(phi, tau_step, t1);
  for (auto& p : advected_parts) {
    measure(p.sampler(), phi, t0, t1, local_neighborhood, local_errors);
  }
  local_neighborhood.vertices().iterate_indices(
      [&](auto const... is) { local_errors(is...) /= size(advected_parts); },
      execution_policy::parallel);
  local_neighborhood.write("error_measure_dg.vtk");
}
