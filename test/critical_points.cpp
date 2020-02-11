#include <catch2/catch.hpp>
#include <tatooine/critical_points.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("critical_points", "[critical_points]") {
  grid_sampler<double, 2, vec<double, 2>, interpolation::linear,
               interpolation::linear>
      sampler(2, 2);
  sampler[0][0] = {3.0 / 4.0, 3.0 / 4.0};
  sampler[1][0] = {-9.0 / 4.0, -1.0 / 4.0};
  sampler[0][1] = {-1.0 / 4.0, -9.0 / 4.0};
  sampler[1][1] = {3.0 / 4.0, 3.0 / 4.0};
  REQUIRE(sampler[0][0](0) == 3.0 / 4.0);
  REQUIRE(sampler[0][0](1) == 3.0 / 4.0);
  REQUIRE(sampler[1][0](0) == -9.0 / 4.0);
  REQUIRE(sampler[1][0](1) == -1.0 / 4.0);
  REQUIRE(sampler[0][1](0) == -1.0 / 4.0);
  REQUIRE(sampler[0][1](1) == -9.0 / 4.0);
  REQUIRE(sampler[1][1](0) == 3.0 / 4.0);
  REQUIRE(sampler[1][1](1) == 3.0 / 4.0);
  auto critical_points = find_critical_points(sampler);
  REQUIRE(critical_points.size() == 2);
  REQUIRE((approx_equal(critical_points[0], vec{0.25, 0.25}) ||
           approx_equal(critical_points[0], vec{0.75, 0.75})));
  REQUIRE((approx_equal(critical_points[1], vec{0.25, 0.25}) ||
           approx_equal(critical_points[1], vec{0.75, 0.75})));
  REQUIRE(!approx_equal(critical_points[0], critical_points[1]));
}
//==============================================================================
}
//==============================================================================
