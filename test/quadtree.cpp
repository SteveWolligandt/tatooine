#include <tatooine/quadtree.h>
#include <tatooine/pointset.h>
#include <tatooine/vtk_legacy.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("quadtree", "[quadtree]") {
  quadtree<double>    qt{vec2::zeros(), vec2::ones() * 4, 2};
  pointset<double, 2> ps;

  REQUIRE(qt.bottom_left() == nullptr);

  qt.insert_vertex(ps, ps.insert_vertex(0.5, 0.5).i);
  REQUIRE(qt.bottom_left() == nullptr);
  REQUIRE(qt.num_vertex_indices() == 1);

  qt.insert_vertex(ps, ps.insert_vertex(1.5, 0.5).i);
  REQUIRE(qt.bottom_left() != nullptr);
  REQUIRE(qt.num_vertex_indices() == 0);
  REQUIRE(qt.bottom_left()->num_vertex_indices() == 0);
  REQUIRE(qt.bottom_right()->num_vertex_indices() == 0);
  REQUIRE(qt.top_left()->num_vertex_indices() == 0);
  REQUIRE(qt.top_right()->num_vertex_indices() == 0);
  REQUIRE(qt.bottom_left()->bottom_left()->num_vertex_indices() == 1);
  REQUIRE(qt.bottom_left()->bottom_right()->num_vertex_indices() == 1);

  qt.insert_vertex(ps, ps.insert_vertex(0.75, 0.5).i);
  REQUIRE(qt.bottom_left()->bottom_left()->num_vertex_indices() == 2);

  qt.insert_vertex(ps, ps.insert_vertex(2, 2).i);
  REQUIRE(qt.bottom_left()->top_right()->num_vertex_indices() == 1);
  REQUIRE(qt.bottom_right()->num_vertex_indices() == 1);
  REQUIRE(qt.top_left()->num_vertex_indices() == 1);
  REQUIRE(qt.top_right()->num_vertex_indices() == 1);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
