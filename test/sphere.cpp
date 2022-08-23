#include <tatooine/geometry/sphere.h>
#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::geometry::test {
//==============================================================================
TEST_CASE("sphere_ray_intersection", "[sphere][intersection][ray]") {
  sphere<double, 3> s;
  ray r{vec{-2.0, 0.0, 0.0}, vec{1.0, 0.0, 0.0}};
  auto intersection = s.check_intersection(r);
  REQUIRE(intersection);
  REQUIRE(approx_equal(intersection->position, vec3{-1.0, 0.0, 0.0}));
  discretize(s,5).write_vtp("discretized_sphere.vtp");
}
//==============================================================================
TEST_CASE("sphere_discretization", "[sphere][discretization]") {
  sphere<double, 3> s1{1.0};
  discretize(s1).write_vtk("discretized_sphere_r1.vtk");
  sphere<double, 3> s2{2.0};
  discretize(s2, 3).write_vtk("discretized_sphere_r2.vtk");
  sphere<double, 3> s3{3.0, vec3{4,4,4}};
  discretize(s3, 4).write_vtk("discretized_sphere_r3.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
