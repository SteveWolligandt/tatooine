#include <tatooine/naive_flowmap_discretization.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/numerical_flowmap.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto v   = analytical::fields::numerical::doublegyre{};
  auto phi = naive_flowmap_discretization<real_t, 2>{
      flowmap(v), 0, 10, vec2{0, 0}, vec2{2, 1}, 200, 100};
  phi.forward_mesh().write_vtk(
      "doublegyre_naive_flowmap_discretization_forward.vtk");
  phi.backward_mesh().write_vtk(
      "doublegyre_naive_flowmap_discretization_backward.vtk");
}
