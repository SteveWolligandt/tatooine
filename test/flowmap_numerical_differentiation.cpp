#include <catch2/catch_test_macros.hpp>
#include <tatooine/analytical/doublegyre.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/interactive.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("flowmap_numerical_differentiation",
          "[flowmap][numerical][differentiation]") {
  auto const d_phi = diff(flowmap(analytical::numerical::doublegyre{}));
  auto g = rectilinear_grid{linspace{0.0, 2.0, 200}, linspace{0.0, 1.0, 100}};
  //auto const x0 = g.vertex_at(20, 59);
  auto const t0 = 0;
  auto const tau = 1;
  //auto const phi_sample = d_phi.flowmap()(x0, t0, tau);
  //auto const d_phi_sample = d_phi(x0, t0, tau);
  //CAPTURE(phi_sample, d_phi_sample);
  g.sample_to_vertex_property(
      [&](auto const& x) {
        auto phi = d_phi.flowmap();
        return phi(x, t0, tau);
      },
      "phi", execution_policy::parallel);
  //g.sample_to_vertex_property(
  //    [&](auto const& x) {
  //      auto copy = d_phi;
  //      return copy(x, t0, tau);
  //    },
  //    "d_phi", execution_policy::parallel);
  rendering::interactive::show(g);
  REQUIRE(false);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
