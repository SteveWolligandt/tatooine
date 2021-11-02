#include <tatooine/structured_grid.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(structured_grid3, "unstructured_grid_3",
                 "[unstructured_grid][3d]") {
  read("/home/steve/firetec/valley_losAlamos/output.1000.vts");
}
//==============================================================================
TEST_CASE_METHOD(structured_grid2, "unstructured_grid_2",
                 "[unstructured_grid][2d]") {
  resize(2, 2);
  vertex_at(0, 0) = {0.0, 0.0};
  vertex_at(1, 0) = {3.0, 2.0};
  vertex_at(0, 1) = {1.0, 4.0};
  vertex_at(1, 1) = {4.0, 4.0};

  {
    auto const coords = local_cell_coordinates(vec{0.0, 0.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0));
    REQUIRE(coords(1) == Approx(0));
  }
  {
    auto const coords = local_cell_coordinates(vec{4.0, 4.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(1));
    REQUIRE(coords(1) == Approx(1));
  }
  {
    auto const coords = local_cell_coordinates(vec{3.0, 2.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(1));
    REQUIRE(coords(1) == Approx(0));
  }
  {
    auto const coords = local_cell_coordinates(vec{1.0, 4.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0));
    REQUIRE(coords(1) == Approx(1));
  }
  {
    auto const coords = local_cell_coordinates(vec{0.5, 2.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0));
    REQUIRE(coords(1) == Approx(0.5));
  }
  {
    auto const coords = local_cell_coordinates(vec{3.5, 3.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(1));
    REQUIRE(coords(1) == Approx(0.5));
  }
  {
    auto const coords = local_cell_coordinates(vec{1.5, 1.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.5));
    REQUIRE(coords(1) == Approx(0));
  }
  {
    auto const coords = local_cell_coordinates(vec{2.5, 4.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.5));
    REQUIRE(coords(1) == Approx(1));
  }
  {
    auto const coords = local_cell_coordinates(vec{2.0, 2.5}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.5));
    REQUIRE(coords(1) == Approx(0.5));
  }
  {
    auto const coords = local_cell_coordinates(vec{2.0, 2.5}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.5));
    REQUIRE(coords(1) == Approx(0.5));
  }
  {
    auto const coords = local_cell_coordinates(vec{3.0, 3.375}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.75));
    REQUIRE(coords(1) == Approx(0.75));
  }
  {
    auto const coords = local_cell_coordinates(vec{1.0, 1.375}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.25));
    REQUIRE(coords(1) == Approx(0.25));
  }
  {
    auto const coords = local_cell_coordinates(vec{1.5, 3.125}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.25));
    REQUIRE(coords(1) == Approx(0.75));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
