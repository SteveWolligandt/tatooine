#include <tatooine/celltree.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/unstructured_tetrahedral_grid.h>
#include <tatooine/unstructured_triangular_grid.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("celltree_node_copy", "[celltree][node][copy]") {
  auto mesh      = unstructured_tetrahedral_grid3{};
  auto hierarchy   = celltree{mesh, vec3::zeros(), vec3::ones()};
  auto n           = decltype(hierarchy)::node_type{};
  n.type.leaf.size = 100;
  auto copy        = decltype(hierarchy)::node_type{n};
  REQUIRE(n.type.leaf.size == copy.type.leaf.size);
}
//==============================================================================
TEST_CASE("celltree_2d_triangle", "[celltree][2d][triangles][construction]") {
  auto const num_vertices = std::size_t(1000);
  auto const num_queries  = std::size_t(1000);
  auto const domain       = aabb2{vec2{0, 0}, vec2{1, 1}};

  auto mesh = unstructured_triangular_grid2{};
  for (std::size_t i = 0; i < num_vertices; ++i) {
    mesh.insert_vertex(domain.random_point());
  }
  mesh.build_delaunay_mesh();
  mesh.write_vtk("celltree_construction.vtk");

  auto hierarchy = celltree{mesh};
  for (size_t i = 0; i < num_queries; ++i) {
    REQUIRE(hierarchy.cells_at(domain.random_point()).size() <= 1);
  }
}
//==============================================================================
TEST_CASE("celltree_3d_triangle", "[celltree][3d][triangles][construction]") {
  auto const sphere    = geometry::sphere3{1};
  auto const mesh      = discretize(sphere, 2);
  auto const hierarchy = celltree{mesh};
  hierarchy.check_intersection(ray{vec3::ones(), vec3::zeros()});
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
