#include <catch2/catch_test_macros.hpp>
#include <tatooine/critical_points.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("critical_points", "[critical_points]") {
  auto  dim = std::array{0.0, 1.0};
  auto  g   = rectilinear_grid{dim, dim};
  auto& s   = g.vec2_vertex_property("sampler");
  s(0, 0)   = {3.0 / 4.0, 3.0 / 4.0};
  s(1, 0)   = {-9.0 / 4.0, -1.0 / 4.0};
  s(0, 1)   = {-1.0 / 4.0, -9.0 / 4.0};
  s(1, 1)   = {3.0 / 4.0, 3.0 / 4.0};
  auto critical_points =
      find_critical_points(s.sampler<interpolation::linear>());
  REQUIRE(size(critical_points) == 2);
  REQUIRE((approx_equal(critical_points[0], vec{0.25, 0.25}) ||
           approx_equal(critical_points[0], vec{0.75, 0.75})));
  REQUIRE((approx_equal(critical_points[1], vec{0.25, 0.25}) ||
           approx_equal(critical_points[1], vec{0.75, 0.75})));
  REQUIRE(!approx_equal(critical_points[0], critical_points[1]));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
