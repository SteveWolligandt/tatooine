#include <tatooine/agranovsky_flowmap_discretization.h>
#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/real.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto v   = analytical::numerical::doublegyre{};
  auto phi = flowmap(v);
  phi.use_caching(false);
  auto const t0             = real_number(0);
  auto const t1             = real_number(20);
  auto const delta_t        = real_number(0.1);
  auto const phi_agranovsky = agranovsky_flowmap_discretization<real_number, 2>{
      phi, t0, t1, delta_t, vec2{0, 0}, vec2{2, 1}, 100, 50};
  std::size_t cnt = 0;
  for (auto const& step : phi_agranovsky.steps()) {
    step.grid(forward).write(
        "agranovsky_flowmap_discretization.forward.example." +
        std::to_string(cnt) + ".vtr");
    step.grid(backward).write(
        "agranovsky_flowmap_discretization.backward.example." +
        std::to_string(cnt) + ".vtr");
    ++cnt;
  }

  auto g  = rectilinear_grid{linspace{0.0, 2.0, 1001}, linspace{0.0, 1.0, 501}};
}
