#include <catch2/catch.hpp>
#include <tatooine/simplex_mesh.h>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("simplex_mesh_triangular_2d", "[simplex_mesh][triangular_mesh][2d]"){
  auto       mesh = triangular_mesh_2{};
  auto const v1   = mesh.insert_vertex(vec2{0, 0});
  auto const v2   = mesh.insert_vertex(vec2{1, 0});
  auto const v3   = mesh.insert_vertex(vec2{0, 1});
  auto const c1   = mesh.insert_cell(v1, v2, v3);
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
TEST_CASE("simplex_mesh_tetrahedral_3d", "[simplex_mesh][tetrahedral_mesh][3d]"){
  auto       mesh = tetrahedral_mesh_3{};
  auto const v1   = mesh.insert_vertex(vec3{0, 0, 0});
  auto const v2   = mesh.insert_vertex(vec3{1, 0, 0});
  auto const v3   = mesh.insert_vertex(vec3{0, 1, 0});
  auto const v4   = mesh.insert_vertex(vec3{0, 0, 1});
  auto const c1   = mesh.insert_cell(v1, v2, v3, v4);
  auto const[v1_, v2_, v3_, v4_] = mesh[c1];
  REQUIRE(v1 == v1_);
  REQUIRE(v2 == v2_);
  REQUIRE(v3 == v3_);
  REQUIRE(v4 == v4_);
  REQUIRE(typeid(v1_) == typeid(decltype(mesh)::vertex_handle&));
  REQUIRE(typeid(v2_) == typeid(decltype(mesh)::vertex_handle&));
  REQUIRE(typeid(v3_) == typeid(decltype(mesh)::vertex_handle&));
  REQUIRE(typeid(v4_) == typeid(decltype(mesh)::vertex_handle&));
  auto const [cv1_, cv2_, cv3_, cv4_] = static_cast<decltype(mesh) const&>(mesh)[c1];
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
