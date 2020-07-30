#include <tatooine/triangular_mesh.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("triangular_mesh_copy", "[triangular_mesh][copy]"){
  triangular_mesh mesh;
  auto const       v0 = mesh.insert_vertex(0.0, 0.0, 0.0);
  auto const       v1 = mesh.insert_vertex(1.0, 0.0, 0.0);
  auto const       v2 = mesh.insert_vertex(0.0, 1.0, 0.0);
  auto const       f0 = mesh.insert_triangle(v0, v1, v2);

  auto& vertex_prop = mesh.add_vertex_property<double>("vertex_prop");
  vertex_prop[v0] = 0;
  vertex_prop[v1] = 1;
  vertex_prop[v2] = 2;
  auto& tri_prop = mesh.add_triangle_property<double>("tri_prop");
  tri_prop[f0] = 4;

  auto copied_mesh = mesh;

  REQUIRE(mesh[v0] == copied_mesh[v0]);
  REQUIRE(mesh[v1] == copied_mesh[v1]);
  REQUIRE(mesh[v2] == copied_mesh[v2]);
  mesh[v0](0) = 2;
  REQUIRE_FALSE(mesh[v0] == copied_mesh[v0]);

  {
    auto& copied_vertex_prop = copied_mesh.vertex_property<double>("vertex_prop");
    REQUIRE(vertex_prop[v0] == copied_vertex_prop[v0]);
    REQUIRE(vertex_prop[v1] == copied_vertex_prop[v1]);
    REQUIRE(vertex_prop[v2] == copied_vertex_prop[v2]);

    vertex_prop[v0] = 100;
    REQUIRE_FALSE(vertex_prop[v0] == copied_vertex_prop[v0]);

    auto& copied_tri_prop = copied_mesh.triangle_property<double>("tri_prop");
    REQUIRE(tri_prop[f0] == copied_tri_prop[f0]);

    tri_prop[f0] = 10;
    REQUIRE_FALSE(tri_prop[f0] == copied_tri_prop[f0]);
  }

  copied_mesh = mesh;
  {
    auto& copied_vertex_prop = copied_mesh.vertex_property<double>("vertex_prop");
    REQUIRE(mesh[v0] == copied_mesh[v0]);
    REQUIRE(vertex_prop[v0] == copied_vertex_prop[v0]);

    auto& copied_tri_prop = copied_mesh.triangle_property<double>("tri_prop");
    REQUIRE(mesh[f0] == copied_mesh[f0]);
    REQUIRE(tri_prop[f0] == copied_tri_prop[f0]);
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================