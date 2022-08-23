#include <tatooine/separating_axis_theorem.h>
#include <tatooine/axis_aligned_bounding_box.h>
#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("separating_axis_theorem") {
  SECTION("aabb - triangle with intersection") {
    SECTION("point inside aabb") {
      std::vector<vec2> polygon0{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
      std::vector<vec2> polygon1{{0, 0}, {2, 2}, {2, 3}};
      REQUIRE(!has_separating_axis(polygon0, polygon1));
    }
    SECTION("no point intersection") {
      std::vector<vec2> polygon0{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
      std::vector<vec2> polygon1{{1, -1}, {2, 2}, {-1, 2}};
      REQUIRE(!has_separating_axis(polygon0, polygon1));
    }
  }
  SECTION("aabb - triangle without intersection") {
    std::vector<vec2> polygon0{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    std::vector<vec2> polygon1{{1, -1}, {2, 2}, {3, 2}};
    REQUIRE(!has_separating_axis(polygon0, polygon1));
  }
  SECTION("aabb class"){
    aabb<double, 2> unit_bb{vec{-1, -1}, vec{1, 1}};
    REQUIRE(unit_bb.is_simplex_inside(vec2{2, -1}, vec2{2,2}, vec2{-1, 2}));
    REQUIRE(unit_bb.is_simplex_inside(vec2{-2, -1}, vec2{1,2}, vec2{-2, 2}));
    REQUIRE(unit_bb.is_simplex_inside(vec2{-2, 2}, vec2{-2, -1}, vec2{1, -1}));
    REQUIRE(unit_bb.is_simplex_inside(vec2{-1, -2}, vec2{2, -2}, vec2{2, 1}));

    REQUIRE_FALSE(unit_bb.is_simplex_inside(vec2{1, -2}, vec2{2, -2}, vec2{2, -1}));
    REQUIRE_FALSE(unit_bb.is_simplex_inside(vec2{2, 1}, vec2{2, 2}, vec2{1, 2}));
    REQUIRE_FALSE(unit_bb.is_simplex_inside(vec2{-1, 2}, vec2{-2, 2}, vec2{-2, -1}));
    REQUIRE_FALSE(unit_bb.is_simplex_inside(vec2{-2, -1}, vec2{-2, -2}, vec2{-1, -2}));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
