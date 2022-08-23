#include <tatooine/delaunator.h>
#include <catch2/catch_test_macros.hpp>
#include <tatooine/unstructured_triangular_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
TEST_CASE("delaunator_unit_quad", "[delaunator][quad]"){
  auto points = std::vector<vec2>{{-1, 1}, {1, 1}, {1, -1}, {-1, -1}};
  delaunator::Delaunator d{points};

  for (size_t i = 0; i < d.triangles.size(); i += 3) {
    std::cerr << "Triangle points: ["
              << points[d.triangles[i]] << ", "
              << points[d.triangles[i + 1]] << ", "
              << points[d.triangles[i + 2]] << "]\n";
  }
}
//==============================================================================
TEST_CASE("delaunator_random_unstructured_triangular_grid",
          "[delaunator][random][unstructured_triangular_grid]") {
  random::uniform rand{0.0, 10.0};
  auto mesh = unstructured_triangular_grid2 {};
  for (size_t i = 0; i < 100; ++i) {
    mesh.insert_vertex(vec2{rand});
  }

  mesh.triangulate_delaunay();
  mesh.write_vtk("delaunay.vtk");
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
