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
  for (size_t i = 0; i < 100; ++i) {
    ps.insert_vertex(aabb2{vec2{0, 0}, vec2{1, 1}}.random_point());
  }
  kdtree              hierarchy{ps};
  ps.write_vtk("kdtree2d_vertices.vtk");
  hierarchy.write_vtk("kdtree2d.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
