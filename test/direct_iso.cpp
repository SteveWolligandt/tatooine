#include <catch2/catch.hpp>
#include <tatooine/grid.h>
#include <tatooine/direct_iso.h>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("direct_iso_grid_vertex_sampler") {
  auto g =
      grid{linspace{0.0, 2.0, 3}, linspace{0.0, 2.0, 3}, linspace{0.0, 2.0, 3}};
  auto r = ray{vec3{-0.5, 0.0, 0.5}, vec3{1.0, 1.0, 0.0}};
  direct_iso(r, g);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
