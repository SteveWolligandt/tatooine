#include <tatooine/structured_grid.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(structured_grid3, "structured_grid_3",
                 "[structured_grid][3d]") {
  read("/home/steve/firetec/valley_losAlamos/output.1000.vts");
  auto const aabb  = axis_aligned_bounding_box();
  auto discretized = rectilinear_grid{linspace{aabb.min(0), aabb.max(0), 100},
                                      linspace{aabb.min(1), aabb.max(1), 100},
                                      linspace{aabb.min(2), aabb.max(2), 100}};
  discretize(linear_vertex_property_sampler<real_t>("O2"), discretized, "prop",
             tag::parallel);
  discretized.write_vtk("resampled_O2.vtk");
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
TEST_CASE_METHOD(structured_grid2, "structured_grid_2_linear_sampler",
                 "[structured_grid][2d][linear][sampler]") {
  resize(3, 2);
  vertex_at(0, 0) = {0.0, 0.0};
  vertex_at(1, 0) = {3.0, 2.0};
  vertex_at(0, 1) = {1.0, 4.0};
  vertex_at(1, 1) = {4.0, 4.0};
  vertex_at(2, 0) = {6.0, 3.0};
  vertex_at(2, 1) = {5.0, 5.0};

  auto& prop                             = scalar_vertex_property("prop");
  prop[vertex_handle{plain_index(0, 0)}] = 1;
  prop[vertex_handle{plain_index(1, 0)}] = 2;
  prop[vertex_handle{plain_index(1, 1)}] = 3;
  prop[vertex_handle{plain_index(0, 1)}] = 4;
  prop[vertex_handle{plain_index(2, 0)}] = 5;
  prop[vertex_handle{plain_index(2, 1)}] = 6;
  auto const aabb                        = axis_aligned_bounding_box();
  auto discretized = rectilinear_grid{linspace{aabb.min(0), aabb.max(0), 1000},
                                      linspace{aabb.min(1), aabb.max(1), 1000}};
  discretize(linear_vertex_property_sampler<real_t>("prop"), discretized,
             "prop");
  discretized.write_vtk("resampled_structured_grid.vtk");
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
