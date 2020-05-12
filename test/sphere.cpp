#include <tatooine/geometry/sphere.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::geometry::test {
//==============================================================================
TEST_CASE("sphere", "[sphere]") {
  sphere<double, 3> s;
  ray r{vec{-2.0, 0.0, 0.0}, vec{1.0, 0.0, 0.0}};
  auto intersection = s.check_intersection(r);
  REQUIRE(intersection);
  REQUIRE(approx_equal(intersection->position, vec3{-1.0, 0.0, 0.0}));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
