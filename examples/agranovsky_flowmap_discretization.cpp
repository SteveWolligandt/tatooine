#include <tatooine/agranovsky_flowmap_discretization.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/real.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto v   = analytical::fields::numerical::doublegyre{};
  auto phi = agranovsky_flowmap_discretization<real_t, 2>{
      flowmap(v), 0, 10, 1, vec2{0, 0}, vec2{2, 1}, 200, 100};
  size_t cnt = 0;
  for (auto const& step : phi.steps()) {
    step.forward_mesh().write_vtk(
        "doublegyre_agranoksy_flowmap_discretization_forward" +
        std::to_string(cnt) + ".vtk");
    step.backward_mesh().write_vtk(
        "doublegyre_agranoksy_flowmap_discretization_backward" +
        std::to_string(cnt) + ".vtk");
    ++cnt;
  }
}
