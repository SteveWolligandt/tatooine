#include <tatooine/structured_grid.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(structured_grid3, "structured_grid_3",
                 "[structured_grid][3d]") {
  // read("/home/steve/firetec/valley_losAlamos/output.1000.vts");
}
//==============================================================================
TEST_CASE_METHOD(structured_grid2, "structured_grid_2_cell_coordinates",
                 "[structured_grid][2d][cell_coordinates]") {
  resize(2, 2);
  vertex_at(0, 0) = {0.0, 0.0};
  vertex_at(1, 0) = {3.0, 2.0};
  vertex_at(0, 1) = {1.0, 4.0};
  vertex_at(1, 1) = {4.0, 4.0};
  auto approx_0   = Approx(0).margin(1e-10);
  SECTION("0,0") {
    auto const coords = local_cell_coordinates(vec{0.0, 0.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == approx_0);
    REQUIRE(coords(1) == approx_0);
  }
  SECTION("1,1") {
    auto const coords = local_cell_coordinates(vec{4.0, 4.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(1));
    REQUIRE(coords(1) == Approx(1));
  }
  SECTION("1,0") {
    auto const coords = local_cell_coordinates(vec{3.0, 2.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(1));
    REQUIRE(coords(1) == approx_0);
  }
  SECTION("0,1") {
    auto const coords = local_cell_coordinates(vec{1.0, 4.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == approx_0);
    REQUIRE(coords(1) == Approx(1));
  }
  SECTION("0,0.5") {
    auto const coords = local_cell_coordinates(vec{0.5, 2.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == approx_0);
    REQUIRE(coords(1) == Approx(0.5));
  }
  SECTION("1,0.5") {
    auto const coords = local_cell_coordinates(vec{3.5, 3.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(1));
    REQUIRE(coords(1) == Approx(0.5));
  }
  SECTION("0.5,0") {
    auto const coords = local_cell_coordinates(vec{1.5, 1.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.5));
    REQUIRE(coords(1) == approx_0);
  }
  SECTION("0.5,1") {
    auto const coords = local_cell_coordinates(vec{2.5, 4.0}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.5));
    REQUIRE(coords(1) == Approx(1));
  }
  SECTION("0.5,0.5") {
    auto const coords = local_cell_coordinates(vec{2.0, 2.5}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.5));
    REQUIRE(coords(1) == Approx(0.5));
  }
  SECTION("0.5,0.5") {
    auto const coords = local_cell_coordinates(vec{2.0, 2.5}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.5));
    REQUIRE(coords(1) == Approx(0.5));
  }
  SECTION("0.75,0.75") {
    auto const coords = local_cell_coordinates(vec{3.0, 3.375}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.75));
    REQUIRE(coords(1) == Approx(0.75));
  }
  SECTION("0.25,0.25") {
    auto const coords = local_cell_coordinates(vec{1.0, 1.375}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.25));
    REQUIRE(coords(1) == Approx(0.25));
  }
  SECTION("0.25,0.75") {
    auto const coords = local_cell_coordinates(vec{1.5, 3.125}, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(0.25));
    REQUIRE(coords(1) == Approx(0.75));
  }
}
//==============================================================================
TEST_CASE_METHOD(structured_grid3, "structured_grid_3_cell_coordinates",
                 "[structured_grid][3d][cell_coordinates]") {
  resize(2, 2, 2);
  vertex_at(0, 0, 0) = {0.0, 0.0, 0.0};
  vertex_at(1, 0, 0) = {3.0, 2.0, 0.0};
  vertex_at(0, 1, 0) = {1.0, 4.0, 0.0};
  vertex_at(1, 1, 0) = {4.0, 4.0, 0.0};
  vertex_at(0, 0, 1) = {0.0, 1.0, 1.0};
  vertex_at(1, 0, 1) = {3.0, 3.0, 1.0};
  vertex_at(0, 1, 1) = {1.0, 5.0, 1.0};
  vertex_at(1, 1, 1) = {4.0, 5.0, 1.0};
  auto approx_0      = Approx(0).margin(1e-10);
  SECTION("0,0,0") {
    auto const coords = local_cell_coordinates(vec{0.0, 0.0, 0.0}, 0, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == approx_0);
    REQUIRE(coords(1) == approx_0);
    REQUIRE(coords(2) == approx_0);
  }
  SECTION("1,1,1") {
    auto const coords = local_cell_coordinates(vec{4.0, 5.0, 1.0}, 0, 0, 0);
    CAPTURE(coords);
    REQUIRE(coords(0) == Approx(1));
    REQUIRE(coords(1) == Approx(1));
    REQUIRE(coords(2) == Approx(1));
  }
  SECTION("some point") {
    auto const x      = vec{1.1, 1.7, 0.5};
    auto const coords = local_cell_coordinates(x, 0, 0, 0);
    auto const u      = coords(0);
    auto const v      = coords(1);
    auto const w      = coords(2);
    CAPTURE(coords);
    REQUIRE(0 <= u);
    REQUIRE(u <= 1);
    REQUIRE(0 <= v);
    REQUIRE(v <= 1);
    REQUIRE(0 <= w);
    REQUIRE(w <= 1);
    REQUIRE(approx_equal((1 - u) * (1 - v) * (1 - w) * vertex_at(0, 0, 0) +
                             u * (1 - v) * (1 - w) * vertex_at(1, 0, 0) +
                             (1 - u) * v * (1 - w) * vertex_at(0, 1, 0) +
                             u * v * (1 - w) * vertex_at(1, 1, 0) +
                             (1 - u) * (1 - v) * w * vertex_at(0, 0, 1) +
                             u * (1 - v) * w * vertex_at(1, 0, 1) +
                             (1 - u) * v * w * vertex_at(0, 1, 1) +
                             u * v * w * vertex_at(1, 1, 1),
                         x));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
