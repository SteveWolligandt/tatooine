#include <tatooine/axis_aligned_bounding_box.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("axis_aligned_bounding_box_ray_intersection",
          "[axis_aligned_bounding_box][ray][intersection]") {
  axis_aligned_bounding_box bb{vec{-1.0, -1.0}, vec{1.0, 1.0}};
  {
    ray  r{vec{0.0, 2.0}, vec{0.0, -1.0}};
    auto i = bb.check_intersection(r);
    REQUIRE(i);
    REQUIRE(i->position == vec{0.0, 1.0});
  }
  {
    ray  r{vec{0.0, -2.0}, vec{0.0, 1.0}};
    auto i = bb.check_intersection(r);
    REQUIRE(i);
    REQUIRE(i->position == vec{0.0, -1.0});
  }
  {
    ray  r{vec{-2.0, -1.0}, vec{1.0, 1.0}};
    auto i = bb.check_intersection(r);
    REQUIRE(i);
    REQUIRE(i->position == vec{-1.0, 0.0});
  }
}
//==============================================================================
TEST_CASE(
    "axis_aligned_bounding_box_tetrahderon_3d_inside",
    "[axis_aligned_bounding_box][tetrahedron][simplex][3d][3D][intersection]") {
  axis_aligned_bounding_box bb{vec3::zeros(), vec3::ones()};
  REQUIRE(bb.is_simplex_inside(vec3{0.1, 0.1, 0.1}, vec3{0.2, 0.3, 0.1},
                                   vec3{0.9, 0.8, 0.2}, vec3{0.2, 0.9, 0.8}));
  REQUIRE(bb.is_simplex_inside(vec3{-10, -10, -10}, vec3{10, -10, -10},
                                   vec3{-10, 10, -10}, vec3{0, 0, 10}));
  REQUIRE(bb.is_simplex_inside(vec3{0.1, 0.1, 0.1}, vec3{1.2, 1.3, 1.1},
                                   vec3{1.9, 1.8, 1.2}, vec3{1.2, 1.9, 1.8}));
  REQUIRE_FALSE(
      bb.is_simplex_inside(vec3{1.1, 1.1, 1.1}, vec3{1.2, 1.3, 1.1},
                               vec3{1.9, 1.8, 1.2}, vec3{1.2, 1.9, 1.8}));
  REQUIRE_FALSE(
      bb.is_simplex_inside(-vec3{1.1, 1.1, 1.1}, -vec3{1.2, 1.3, 1.1},
                               -vec3{1.9, 1.8, 1.2}, -vec3{1.2, 1.9, 1.8}));
}
//==============================================================================
TEST_CASE("axis_aligned_bounding_box_rectangle_inside",
          "[axis_aligned_bounding_box][rectangle][intersection]") {
  axis_aligned_bounding_box bb{vec2::zeros(), vec2::ones()};
  REQUIRE_FALSE(bb.is_rectangle_inside(vec2{1.5, 1.0}, vec2{2.0, 1.5},
                                       vec2{1.0, 2.5}, vec2{0.5, 2.0}));
  REQUIRE(bb.is_rectangle_inside(vec2{1.5, 0.0}, vec2{2.0, 0.5},
                                       vec2{1.0, 1.5}, vec2{0.5, 1.0}));
}
//==============================================================================
TEST_CASE("axis_aligned_bounding_box_triangle_2d_inside",
          "[axis_aligned_bounding_box][triangle][simplex][2d][2D][intersection]") {
  axis_aligned_bounding_box bb{vec2::zeros(), vec2::ones()};
  REQUIRE_FALSE(
      bb.is_simplex_inside(vec2{2.0, 0.5}, vec2{1.5, 2.0}, vec2{0.5, 2.0}));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
