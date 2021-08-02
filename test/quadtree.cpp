#include <tatooine/quadtree.h>
#include <tatooine/unstrucutured_triangular_grid.h>
#include <tatooine/vtk_legacy.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("quadtree", "[quadtree]") {
  quadtree<double>    qt{vec2::zeros(), vec2::ones() * 4, 2};
  unstructured_triangular_grid_2 mesh;

  REQUIRE_FALSE(qt.is_splitted());
  REQUIRE_FALSE(qt.holds_vertices());

  auto t0 = mesh.insert_vertex(0.5, 0.5);
  qt.insert_vertex(mesh, t0.i);
  REQUIRE_FALSE(qt.is_splitted());
  REQUIRE(qt.holds_vertices());

  qt.insert_vertex(mesh, mesh.insert_vertex(1.5, 0.5).i);
  REQUIRE(qt.is_splitted());
  REQUIRE_FALSE(qt.holds_vertices());
  REQUIRE_FALSE(qt.bottom_left()->holds_vertices());
  REQUIRE_FALSE(qt.bottom_right()->holds_vertices());
  REQUIRE_FALSE(qt.top_left()->holds_vertices());
  REQUIRE_FALSE(qt.top_right()->holds_vertices());
  REQUIRE(qt.bottom_left()->bottom_left()->num_vertex_handles() == 1);
  REQUIRE(qt.bottom_left()->bottom_right()->num_vertex_handles() == 1);

  qt.insert_vertex(mesh, mesh.insert_vertex(0.75, 0.5).i);
  REQUIRE(qt.bottom_left()->bottom_left()->num_vertex_handles() == 2);

  qt.insert_vertex(mesh, mesh.insert_vertex(2, 2).i);
  REQUIRE(qt.bottom_left()->top_right()->num_vertex_handles() == 1);
  REQUIRE(qt.bottom_right()->num_vertex_handles() == 1);
  REQUIRE(qt.top_left()->num_vertex_handles() == 1);
  REQUIRE(qt.top_right()->num_vertex_handles() == 1);

  auto t1 = mesh.insert_vertex(1.75, 0.5);
  qt.insert_vertex(mesh, t1.i);
  REQUIRE(qt.bottom_left()->bottom_left()->num_vertex_handles() == 2);
  REQUIRE(qt.bottom_left()->bottom_right()->num_vertex_handles() == 2);
  REQUIRE(qt.bottom_left()->top_left()->num_vertex_handles() == 0);
  REQUIRE(qt.bottom_left()->top_right()->num_vertex_handles() == 1);

  auto t2 = mesh.insert_vertex(0.5, 1.75);
  qt.insert_vertex(mesh, t2.i);
  REQUIRE(qt.bottom_left()->bottom_left()->num_vertex_handles() == 2);
  REQUIRE(qt.bottom_left()->bottom_right()->num_vertex_handles() == 2);
  REQUIRE(qt.bottom_left()->top_left()->num_vertex_handles() == 1);
  REQUIRE(qt.bottom_left()->top_right()->num_vertex_handles() == 1);

  qt.insert_face(mesh, mesh.insert_face(t0, t1, t2).i);
  REQUIRE(qt.bottom_left()->bottom_left()->num_vertex_handles() == 2);
  REQUIRE(qt.bottom_left()->bottom_right()->num_vertex_handles() == 2);
  REQUIRE(qt.bottom_left()->top_left()->num_vertex_handles() == 1);
  REQUIRE(qt.bottom_left()->top_right()->num_vertex_handles() == 1);
  REQUIRE(qt.bottom_left()->bottom_left()->num_face_handles() == 1);
  REQUIRE(qt.bottom_left()->bottom_right()->num_face_handles() == 1);
  REQUIRE(qt.bottom_left()->top_left()->num_face_handles() == 1);
  REQUIRE(qt.bottom_left()->top_right()->num_face_handles() == 1);
  REQUIRE_FALSE(qt.bottom_right()->is_splitted());
  REQUIRE_FALSE(qt.top_right()->is_splitted());
  REQUIRE_FALSE(qt.top_left()->is_splitted());
  REQUIRE_FALSE(qt.bottom_right()->holds_faces());
  REQUIRE_FALSE(qt.top_right()->holds_faces());
  REQUIRE_FALSE(qt.top_left()->holds_faces());

  // face get
  REQUIRE(qt.nearby_faces(vec2{0.5, 0.5}).size() == 1);
  REQUIRE(qt.nearby_faces(vec2{1.5, 0.5}).size() == 1);
  REQUIRE(qt.nearby_faces(vec2{2.5, 0.5}).size() == 0);
  REQUIRE(qt.nearby_faces(vec2{3.5, 0.5}).size() == 0);

  REQUIRE(qt.nearby_faces(vec2{0.5, 1.5}).size() == 1);
  REQUIRE(qt.nearby_faces(vec2{1.5, 1.5}).size() == 1);
  REQUIRE(qt.nearby_faces(vec2{2.5, 1.5}).size() == 0);
  REQUIRE(qt.nearby_faces(vec2{3.5, 1.5}).size() == 0);

  REQUIRE(qt.nearby_faces(vec2{0.5, 2.5}).size() == 0);
  REQUIRE(qt.nearby_faces(vec2{1.5, 2.5}).size() == 0);
  REQUIRE(qt.nearby_faces(vec2{2.5, 2.5}).size() == 0);
  REQUIRE(qt.nearby_faces(vec2{3.5, 2.5}).size() == 0);

  REQUIRE(qt.nearby_faces(vec2{0.5, 3.5}).size() == 0);
  REQUIRE(qt.nearby_faces(vec2{1.5, 3.5}).size() == 0);
  REQUIRE(qt.nearby_faces(vec2{2.5, 3.5}).size() == 0);
  REQUIRE(qt.nearby_faces(vec2{3.5, 3.5}).size() == 0);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
