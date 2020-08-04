#include <catch2/catch.hpp>
#include <tatooine/critical_points.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("critical_points", "[critical_points]") {
  std::array dim{0.0, 1.0};
  grid<decltype(dim), decltype(dim)> g{dim, dim};
  auto&                              s = g.add_contiguous_vertex_property<
      vec<double, 2>, x_fastest, interpolation::linear, interpolation::linear>(
      "sampler");
  s.container().at(0, 0) = {3.0 / 4.0, 3.0 / 4.0};
  s.container().at(1, 0) = {-9.0 / 4.0, -1.0 / 4.0};
  s.container().at(0, 1) = {-1.0 / 4.0, -9.0 / 4.0};
  s.container().at(1, 1) = {3.0 / 4.0, 3.0 / 4.0};
  REQUIRE(s.data_at(0, 0)(0) == 3.0 / 4.0);
  REQUIRE(s.data_at(0, 0)(1) == 3.0 / 4.0);
  REQUIRE(s.data_at(1, 0)(0) == -9.0 / 4.0);
  REQUIRE(s.data_at(1, 0)(1) == -1.0 / 4.0);
  REQUIRE(s.data_at(0, 1)(0) == -1.0 / 4.0);
  REQUIRE(s.data_at(0, 1)(1) == -9.0 / 4.0);
  REQUIRE(s.data_at(1, 1)(0) == 3.0 / 4.0);
  REQUIRE(s.data_at(1, 1)(1) == 3.0 / 4.0);
  auto critical_points = find_critical_points(s);
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
