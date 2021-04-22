#include <catch2/catch.hpp>
#include <tatooine/kdtree.h>
#include <tatooine/pointset.h>
#include <tatooine/real.h>
#include <tatooine/geometry/sphere.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("kdtree_pointset_2d", "[kdtree][pointset][2d]") {
  pointset<real_t, 2> ps;
  size_t const num_points = 101;
  for (size_t i = 0; i < num_points; ++i) {
    ps.insert_vertex(aabb2{vec2{0, 0}, vec2{2, 1}}.random_point());
  }
  kdtree              hierarchy{ps};
  ps.write_vtk("kdtree2d_vertices.vtk");
  hierarchy.write_vtk("kdtree2d.vtk");
}
//==============================================================================
TEST_CASE("kdtree_pointset_3d", "[kdtree][pointset][3d]") {
  pointset<real_t, 3> ps;
  size_t const num_points = 101;
  for (size_t i = 0; i < num_points; ++i) {
    ps.insert_vertex(aabb3{vec3{0, 0, 0}, vec3{2, 1, 1}}.random_point());
  }
  kdtree              hierarchy{ps};
  ps.write_vtk("kdtree3d_vertices.vtk");
  hierarchy.write_vtk("kdtree3d.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
