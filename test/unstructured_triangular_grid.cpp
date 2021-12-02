#include <tatooine/unstructured_triangular_grid.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid3,
                 "unstructured_triangular_grid_vertex_add") {
  auto v0 = insert_vertex(0,0,0);
  auto v1 = insert_vertex(1,2,3);

  REQUIRE(at(v0)(0) == 0);
  REQUIRE(at(v0)(1) == 0);
  REQUIRE(at(v0)(2) == 0);
  REQUIRE(at(v1)(0) == 1);
  REQUIRE(at(v1)(1) == 2);
  REQUIRE(at(v1)(2) == 3);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
