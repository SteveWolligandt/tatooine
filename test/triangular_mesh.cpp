#include <tatooine/triangular_mesh.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("triangular_mesh_copy", "[triangular_mesh][copy]") {
  triangular_mesh3 mesh;
  auto const       v0 = mesh.insert_vertex(0.0, 0.0, 0.0);
  auto const       v1 = mesh.insert_vertex(1.0, 0.0, 0.0);
  auto const       v2 = mesh.insert_vertex(0.0, 1.0, 0.0);
  auto const       f0 = mesh.insert_cell(v0, v1, v2);

  auto& vertex_prop = mesh.scalar_vertex_property("vertex_prop");
  vertex_prop[v0]   = 0;
  vertex_prop[v1]   = 1;
  vertex_prop[v2]   = 2;
  auto& cell_prop   = mesh.scalar_cell_property("cell_prop");
  cell_prop[f0]     = 4;

  auto copied_mesh = mesh;

  REQUIRE(mesh[v0] == copied_mesh[v0]);
  REQUIRE(mesh[v1] == copied_mesh[v1]);
  REQUIRE(mesh[v2] == copied_mesh[v2]);
  mesh[v0](0) = 2;
  REQUIRE_FALSE(mesh[v0] == copied_mesh[v0]);

  {
    auto& copied_vertex_prop =
        copied_mesh.scalar_vertex_property("vertex_prop");
    REQUIRE(vertex_prop[v0] == copied_vertex_prop[v0]);
    REQUIRE(vertex_prop[v1] == copied_vertex_prop[v1]);
    REQUIRE(vertex_prop[v2] == copied_vertex_prop[v2]);

    vertex_prop[v0] = 100;
    REQUIRE_FALSE(vertex_prop[v0] == copied_vertex_prop[v0]);

    auto& copied_cell_prop = copied_mesh.scalar_cell_property("cell_prop");
    REQUIRE(cell_prop[f0] == copied_cell_prop[f0]);

    cell_prop[f0] = 10;
    REQUIRE_FALSE(cell_prop[f0] == copied_cell_prop[f0]);
  }

  copied_mesh = mesh;
  {
    auto& copied_vertex_prop =
        copied_mesh.scalar_vertex_property("vertex_prop");
    REQUIRE(mesh[v0] == copied_mesh[v0]);
    REQUIRE(vertex_prop[v0] == copied_vertex_prop[v0]);

    auto& copied_cell_prop = copied_mesh.scalar_cell_property("cell_prop");
    REQUIRE(mesh[f0] == copied_mesh[f0]);
    REQUIRE(cell_prop[f0] == copied_cell_prop[f0]);
  }
}
//==============================================================================
TEST_CASE("triangular_mesh_linear_sampler",
          "[triangular_mesh][linear_sampler]") {
  triangular_mesh<double, 2> mesh;
  auto const                 v0 = mesh.insert_vertex(0.0, 0.0);
  auto const                 v1 = mesh.insert_vertex(1.0, 0.0);
  auto const                 v2 = mesh.insert_vertex(0.0, 1.0);
  auto const                 v3 = mesh.insert_vertex(1.0, 1.0);
  mesh.insert_cell(v0, v1, v2);
  mesh.insert_cell(v1, v3, v2);

  auto& prop   = mesh.scalar_vertex_property("prop");
  prop[v0]     = 1;
  prop[v1]     = 2;
  prop[v2]     = 3;
  prop[v3]     = 4;
  auto sampler = mesh.sampler(prop);
  REQUIRE(sampler(mesh[v0]) == prop[v0]);
  REQUIRE(sampler(mesh[v1]) == prop[v1]);
  REQUIRE(sampler(mesh[v2]) == prop[v2]);
  REQUIRE(sampler(mesh[v3]) == prop[v3]);
  REQUIRE(sampler(vec2{0.5, 0.5}) == Approx(2.5));
  REQUIRE(sampler(vec2{0.0, 0.0}) == Approx(1));
  REQUIRE(sampler(vec2{1.0, 1.0}) == Approx(4));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
