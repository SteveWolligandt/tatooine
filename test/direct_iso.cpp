#include <catch2/catch.hpp>
#include <tatooine/grid.h>
#include <tatooine/direct_iso.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("direct_iso_grid_vertex_sampler") {
  auto g =
      grid{linspace{0.0, 2.0, 3}, linspace{0.0, 2.0, 3}, linspace{0.0, 2.0, 3}};
  auto& s       = g.add_scalar_vertex_property("s");
  auto  sampler = s.linear_sampler();
  g.loop_over_vertex_indices([&s](auto const ix, auto const iy, auto const iz) {
    if (iy == 1) {
      s(ix, iy, iz) = -1;
    } else {
      s(ix, iy, iz) = 1;
    }
  });

  rendering::perspective_camera cam{
      vec3{-2, -2, -2}, vec3{1, 1, 1}, vec3{0, 0, 1}, 60, 500, 500};
  direct_iso(cam, sampler, 0);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
