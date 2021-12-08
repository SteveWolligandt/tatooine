#include <tatooine/unstructured_triangular_grid.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(UnstructuredTriangularGrid3<double>,
                 "unstructured_triangular_grid_io",
                 "[unstructured_triangular_grid][triangular_grid][io][IO]") {
  auto v0 = insert_vertex(0, 0, 0);
  auto v1 = insert_vertex(1, 0, 0);
  auto v2 = insert_vertex(0, 1, 0);
  insert_cell(v0, v1, v2);
  write_vtp("triangle_poly.vtp");
}
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid3, "unstructured_triangular_grid_copy",
          "[unstructured_triangular_grid][copy]") {
  auto const                     v0 = insert_vertex(0.0, 0.0, 0.0);
  auto const                     v1 = insert_vertex(1.0, 0.0, 0.0);
  auto const                     v2 = insert_vertex(0.0, 1.0, 0.0);
  auto const                     f0 = insert_cell(v0, v1, v2);

  auto& vertex_prop = scalar_vertex_property("vertex_prop");
  vertex_prop[v0]   = 0;
  vertex_prop[v1]   = 1;
  vertex_prop[v2]   = 2;
  auto& cell_prop   = scalar_cell_property("cell_prop");
  cell_prop[f0]     = 4;

  auto copied_mesh = *this;

  REQUIRE(at(v0) == copied_mesh[v0]);
  REQUIRE(at(v1) == copied_mesh[v1]);
  REQUIRE(at(v2) == copied_mesh[v2]);
  at(v0)(0) = 2;
  REQUIRE_FALSE(at(v0) == copied_mesh[v0]);

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

  copied_mesh = *this;
  {
    auto& copied_vertex_prop =
        copied_mesh.scalar_vertex_property("vertex_prop");
    REQUIRE(at(v0) == copied_mesh[v0]);
    REQUIRE(vertex_prop[v0] == copied_vertex_prop[v0]);

    auto& copied_cell_prop = copied_mesh.scalar_cell_property("cell_prop");
    REQUIRE(at(f0) == copied_mesh[f0]);
    REQUIRE(cell_prop[f0] == copied_cell_prop[f0]);
  }
}
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid2,
                 "unstructured_triangular_grid_linear_sampler",
                 "[unstructured_triangular_grid][linear_sampler]") {
  auto const                     v0 = insert_vertex(0.0, 0.0);
  auto const                     v1 = insert_vertex(1.0, 0.0);
  auto const                     v2 = insert_vertex(0.0, 1.0);
  auto const                     v3 = insert_vertex(1.0, 1.0);
  insert_cell(v0, v1, v2);
  insert_cell(v1, v3, v2);

  auto& prop   = scalar_vertex_property("prop");
  prop[v0]     = 1;
  prop[v1]     = 2;
  prop[v2]     = 3;
  prop[v3]     = 4;
  auto sampler = this->sampler(prop);
  REQUIRE(sampler(at(v0)) == prop[v0]);
  REQUIRE(sampler(at(v1)) == prop[v1]);
  REQUIRE(sampler(at(v2)) == prop[v2]);
  REQUIRE(sampler(at(v3)) == prop[v3]);
  REQUIRE(sampler(vec2{0.5, 0.5}) == Approx(2.5));
  REQUIRE(sampler(vec2{0.0, 0.0}) == Approx(1));
  REQUIRE(sampler(vec2{1.0, 1.0}) == Approx(4));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
