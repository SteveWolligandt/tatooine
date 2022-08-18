#include <catch2/catch.hpp>
#include <tatooine/unstructured_tetrahedral_grid.h>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("unstructured_simplicial_grid_tetrahedral_3d",
          "[unstructured_simplicial_grid][unstructured_tetrahedral_grid][3d]") {
  auto       mesh = unstructured_tetrahedral_grid3{};
  auto const v1   = mesh.insert_vertex(0, 0, 0);
  auto const v2   = mesh.insert_vertex(1, 0, 0);
  auto const v3   = mesh.insert_vertex(0, 1, 0);
  auto const v4   = mesh.insert_vertex(0, 0, 1);
  auto const c1   = mesh.insert_simplex(v1, v2, v3, v4);
  auto const[v1_, v2_, v3_, v4_] = mesh[c1];
  REQUIRE(v1 == v1_);
  REQUIRE(v2 == v2_);
  REQUIRE(v3 == v3_);
  REQUIRE(v4 == v4_);
  REQUIRE(typeid(v1_) == typeid(decltype(mesh)::vertex_handle&));
  REQUIRE(typeid(v2_) == typeid(decltype(mesh)::vertex_handle&));
  REQUIRE(typeid(v3_) == typeid(decltype(mesh)::vertex_handle&));
  REQUIRE(typeid(v4_) == typeid(decltype(mesh)::vertex_handle&));
  auto const [cv1_, cv2_, cv3_, cv4_] =
      static_cast<decltype(mesh) const&>(mesh)[c1];
  REQUIRE(v1 == cv1_);
  REQUIRE(v2 == cv2_);
  REQUIRE(v3 == cv3_);
  REQUIRE(v4 == cv4_);
  REQUIRE(typeid(cv1_) == typeid(decltype(mesh)::vertex_handle const&));
  REQUIRE(typeid(cv2_) == typeid(decltype(mesh)::vertex_handle const&));
  REQUIRE(typeid(cv3_) == typeid(decltype(mesh)::vertex_handle const&));
  REQUIRE(typeid(cv4_) == typeid(decltype(mesh)::vertex_handle const&));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
