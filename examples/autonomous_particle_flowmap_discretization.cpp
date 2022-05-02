#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
using namespace tatooine;
using analytical::fields::numerical::doublegyre;
auto main() -> int {
  auto eps = 1e-10;
  auto v   = doublegyre{};
  auto phi = autonomous_particle_flowmap_discretization2{
      flowmap(v), 0, 2, 0.01,
      rectilinear_grid{linspace{0.0 + eps, 2.0 - eps, 51},
                       linspace{0.0 + eps, 2.0 - eps, 26}}};
  auto resample_grid = rectilinear_grid{linspace{0.0, 2.0, 200}, linspace{0.0, 1.0, 100}};
  resample_grid.sample_to_vertex_property(
      [&](auto const& p) { return phi.sample(p, forward); }, "flowmap_forward");
  resample_grid.sample_to_vertex_property(
      [&](auto const& p) { return phi.sample(p, backward); }, "flowmap_backward");
  resample_grid.write("autonomous_particle_flowmap_discretization.example.vtr");
  phi.pointset(forward).write(
      "autonomous_particle_flowmap_discretization_forward_samples.example.vtp");
  phi.pointset(backward).write(
      "autonomous_particle_flowmap_discretization_backward_samples.example.vtp");
}
