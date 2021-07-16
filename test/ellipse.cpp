#include <tatooine/geometry/ellipse.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("ellipse_is_inside", "[ellipse][is_inside]") {
  auto e = geometry::ellipse{1.0, 2.0};
  discretize(e, 100).write_vtk("ellipse_1_2.vtk");
  REQUIRE(e.is_inside(vec2{0,0}));
  REQUIRE(e.is_inside(vec2{1,0}));
  REQUIRE(e.is_inside(vec2{-1,0}));
  REQUIRE(e.is_inside(vec2{0,1}));
  REQUIRE(e.is_inside(vec2{0,-1}));
  REQUIRE_FALSE(e.is_inside(vec2{1,1}));
  REQUIRE_FALSE(e.is_inside(vec2{1,2}));
  REQUIRE_FALSE(e.is_inside(vec2{-1,2}));
  REQUIRE_FALSE(e.is_inside(vec2{1,-2}));
  REQUIRE_FALSE(e.is_inside(vec2{-1,-2}));
  REQUIRE(e.is_inside(vec2{0.654541,-0.906931}));
  REQUIRE_FALSE(e.is_inside(vec2{1.65192, 0.377992}));
  REQUIRE_FALSE(e.is_inside(vec2{1.43124,-0.790627}));
  REQUIRE(e.is_inside(vec2{0.7133255968905649,0.7194715373288222}));
  REQUIRE_FALSE(e.is_inside(vec2{0.9133308265525608,0.8467475925682734}));
  REQUIRE(e.is_inside(vec2{0.8986451278710821,0.8397544027199515}));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================

