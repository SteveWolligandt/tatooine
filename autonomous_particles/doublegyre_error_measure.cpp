#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>

using namespace tatooine;
using tatooine::analytical::fields::numerical::doublegyre;
auto main() -> int {
  auto v = doublegyre{};
  auto phi = flowmap(v);
  auto t0 = real_t(0);
  auto tau = real_t(3);
  auto tau_step = real_t(0.1);
  auto discretized_flowmap = autonomous_particle_flowmap_discretization{
      phi, t0, tau, tau_step,
      uniform_rectilinear_grid{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}}};
}
