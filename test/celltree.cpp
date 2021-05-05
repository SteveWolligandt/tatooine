#include <catch2/catch.hpp>
#include <tatooine/celltree.h>
#include <tatooine/triangular_mesh.h>
#include <tatooine/tetrahedral_mesh.h>
#include <tatooine/geometry/sphere.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("celltree_node_copy", "[celltree][node][copy]") {
  auto mesh      = tetrahedral_mesh3{};
  auto hierarchy = celltree{mesh};
  decltype(hierarchy)::node_t n;
  n.type.leaf.size = 100;
  decltype(hierarchy)::node_t copy{n};
  REQUIRE(n.type.leaf.size == copy.type.leaf.size);
}
//==============================================================================
TEST_CASE("celltree", "[celltree][construction]") {
  auto const num_vertices = std::size_t(1000);
  auto const num_queries = std::size_t(1000);
  auto const domain       = aabb2{vec2{0, 0}, vec2{1, 1}};

  auto mesh = triangular_mesh2{};
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
}  // namespace tatooine::test
//==============================================================================
