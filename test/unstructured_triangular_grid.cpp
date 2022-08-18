#include <tatooine/unstructured_triangular_grid.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid3,
                 "unstructured_triangular_grid_vertex_add") {
  auto v0 = insert_vertex(0,0,0);
  auto v1 = insert_vertex(1,0,0);
  auto v2 = insert_vertex(0,1,0);
  insert_simplex(v0,v1,v2);
  REQUIRE(at(v0)(0) == 0);
  REQUIRE(at(v0)(1) == 0);
  REQUIRE(at(v0)(2) == 0);
  REQUIRE(at(v1)(0) == 1);
  REQUIRE(at(v1)(1) == 2);
  REQUIRE(at(v1)(2) == 3);
}
//==============================================================================
TEST_CASE("unstructured_simplicial_grid_triangular_2d",
          "[unstructured_simplicial_grid][unstructured_triangular_grid][2d]") {
  auto       mesh = unstructured_triangular_grid2{};
  auto const v1   = mesh.insert_vertex(0, 0);
  auto const v2   = mesh.insert_vertex(1, 0);
  auto const v3   = mesh.insert_vertex(0, 1);
  auto const c1   = mesh.insert_simplex(v1, v2, v3);
  auto const[v1_, v2_, v3_] = mesh[c1];
  REQUIRE(v1 == v1_);
  REQUIRE(v2 == v2_);
  REQUIRE(v3 == v3_);
  REQUIRE(typeid(v1_) == typeid(decltype(mesh)::vertex_handle&));
  REQUIRE(typeid(v2_) == typeid(decltype(mesh)::vertex_handle&));
  REQUIRE(typeid(v3_) == typeid(decltype(mesh)::vertex_handle&));
  auto const [cv1_, cv2_, cv3_] = static_cast<decltype(mesh) const&>(mesh)[c1];
  REQUIRE(v1 == cv1_);
  REQUIRE(v2 == cv2_);
  REQUIRE(v3 == cv3_);
  REQUIRE(typeid(cv1_) == typeid(decltype(mesh)::vertex_handle const&));
  REQUIRE(typeid(cv2_) == typeid(decltype(mesh)::vertex_handle const&));
  REQUIRE(typeid(cv3_) == typeid(decltype(mesh)::vertex_handle const&));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
