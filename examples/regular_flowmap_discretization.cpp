#include <tatooine/regular_flowmap_discretization.h>
#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/numerical_flowmap.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto v   = analytical::fields::numerical::doublegyre{};
  auto phi = regular_flowmap_discretization<real_number, 2>{
      flowmap(v), 0, 10, vec2{0, 0}, vec2{2, 1}, 200, 100};
  auto g = rectilinear_grid{linspace{0.0, 2.0, 1000}, linspace{0.0, 1.0, 500}};
  g.sample_to_vertex_property(
      [&](vec2 const& x) { return phi.sample(x, forward); }, "forward",
      execution_policy::parallel);
  g.sample_to_vertex_property(
      [&](vec2 const& x) { return phi.sample(x, backward); }, "backward",
      execution_policy::parallel);
  g.write("regular_flowmap_discretization.example.vtr");
}
