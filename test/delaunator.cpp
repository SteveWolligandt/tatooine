#include <tatooine/delaunator.h>
#include <catch2/catch.hpp>
#include <tatooine/triangular_mesh.h>
//==============================================================================
namespace tatooine {
//==============================================================================
TEST_CASE("delaunator_unit_quad", "[delaunator][quad]"){
  std::vector<vec2> points{{-1, 1}, {1, 1}, {1, -1}, {-1, -1}};
  delaunator::Delaunator d{points};

  for (size_t i = 0; i < d.triangles.size(); i += 3) {
    std::cerr << "Triangle points: ["
              << points[d.triangles[i]] << ", "
              << points[d.triangles[i + 1]] << ", "
              << points[d.triangles[i + 2]] << "]\n";
  }
}
//==============================================================================
TEST_CASE("delaunator_random_triangular_mesh",
          "[delaunator][random][triangular_mesh]") {
  random_uniform rand{0.0, 10.0};
  triangular_mesh<double, 2> mesh;
  for (size_t i = 0; i < 100; ++i) {
    mesh.insert_vertex(vec<double, 2>{rand});
  }

  mesh.triangulate_delaunay();
  mesh.write_vtk("delaunay.vtk");
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
