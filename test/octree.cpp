#include <tatooine/octree.h>
#include <tatooine/tetrahedral_mesh.h>
#include <tatooine/vtk_legacy.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("octree", "[octree]") {
  octree<double>   ot{vec3::zeros(), vec3::ones() * 4, 2};
  tetrahedral_mesh mesh;

  REQUIRE_FALSE(ot.is_splitted());
  REQUIRE_FALSE(ot.holds_vertices());

  auto v0 = mesh.insert_vertex(0.5, 0.5, 0.5);
  ot.insert_vertex(mesh, v0.i);
  REQUIRE_FALSE(ot.is_splitted());
  REQUIRE(ot.holds_vertices());

  ot.insert_vertex(mesh, mesh.insert_vertex(1.5, 0.5, 0.5).i);
  REQUIRE(ot.is_splitted());
  REQUIRE_FALSE(ot.holds_vertices());
  REQUIRE_FALSE(ot.left_bottom_front()->holds_vertices());
  REQUIRE_FALSE(ot.right_bottom_front()->holds_vertices());
  REQUIRE_FALSE(ot.left_top_front()->holds_vertices());
  REQUIRE_FALSE(ot.right_top_front()->holds_vertices());
  REQUIRE_FALSE(ot.left_bottom_back()->holds_vertices());
  REQUIRE_FALSE(ot.right_bottom_back()->holds_vertices());
  REQUIRE_FALSE(ot.left_top_back()->holds_vertices());
  REQUIRE_FALSE(ot.right_top_back()->holds_vertices());

  REQUIRE(ot.left_bottom_front()->left_bottom_front()->num_vertex_handles() == 1);
  REQUIRE(ot.left_bottom_front()->right_bottom_front()->num_vertex_handles() == 1);

  ot.insert_vertex(mesh, mesh.insert_vertex(0.75, 0.5, 0.5).i);
  REQUIRE(ot.left_bottom_front()->left_bottom_front()->num_vertex_handles() == 2);

  ot.insert_vertex(mesh, mesh.insert_vertex(2, 2, 2).i);
  REQUIRE(ot.left_bottom_front()->right_top_back()->num_vertex_handles() == 1);
  REQUIRE(ot.right_bottom_front()->num_vertex_handles() == 1);
  REQUIRE(ot.left_top_front()->num_vertex_handles() == 1);
  REQUIRE(ot.right_top_front()->num_vertex_handles() == 1);
  REQUIRE(ot.left_bottom_back()->num_vertex_handles() == 1);
  REQUIRE(ot.right_bottom_back()->num_vertex_handles() == 1);
  REQUIRE(ot.left_top_back()->num_vertex_handles() == 1);
  REQUIRE(ot.right_top_back()->num_vertex_handles() == 1);

  //auto v1 = mesh.insert_vertex(1.75, 0.5);
  //ot.insert_vertex(mesh, v1.i);
  //REQUIRE(ot.left_bottom()->left_bottom()->num_vertex_handles() == 2);
  //REQUIRE(ot.left_bottom()->right_bottom()->num_vertex_handles() == 2);
  //REQUIRE(ot.left_bottom()->left_top()->num_vertex_handles() == 0);
  //REQUIRE(ot.left_bottom()->right_top()->num_vertex_handles() == 1);
  //
  //auto v2 = mesh.insert_vertex(0.5, 1.75);
  //ot.insert_vertex(mesh, v2.i);
  //REQUIRE(ot.left_bottom()->left_bottom()->num_vertex_handles() == 2);
  //REQUIRE(ot.left_bottom()->right_bottom()->num_vertex_handles() == 2);
  //REQUIRE(ot.left_bottom()->left_top()->num_vertex_handles() == 1);
  //REQUIRE(ot.left_bottom()->right_top()->num_vertex_handles() == 1);
  //
  //ot.insert_face(mesh, mesh.insert_face(v0, v1, v2).i);
  //REQUIRE(ot.left_bottom()->left_bottom()->num_vertex_handles() == 2);
  //REQUIRE(ot.left_bottom()->right_bottom()->num_vertex_handles() == 2);
  //REQUIRE(ot.left_bottom()->left_top()->num_vertex_handles() == 1);
  //REQUIRE(ot.left_bottom()->right_top()->num_vertex_handles() == 1);
  //REQUIRE(ot.left_bottom()->left_bottom()->num_face_handles() == 1);
  //REQUIRE(ot.left_bottom()->right_bottom()->num_face_handles() == 1);
  //REQUIRE(ot.left_bottom()->left_top()->num_face_handles() == 1);
  //REQUIRE(ot.left_bottom()->right_top()->num_face_handles() == 1);
  //REQUIRE_FALSE(ot.right_bottom()->is_splitted());
  //REQUIRE_FALSE(ot.right_top()->is_splitted());
  //REQUIRE_FALSE(ot.left_top()->is_splitted());
  //REQUIRE_FALSE(ot.right_bottom()->holds_faces());
  //REQUIRE_FALSE(ot.right_top()->holds_faces());
  //REQUIRE_FALSE(ot.left_top()->holds_faces());
  //
  //// face get
  //REQUIRE(ot.nearby_faces(vec2{0.5, 0.5}).size() == 1);
  //REQUIRE(ot.nearby_faces(vec2{1.5, 0.5}).size() == 1);
  //REQUIRE(ot.nearby_faces(vec2{2.5, 0.5}).size() == 0);
  //REQUIRE(ot.nearby_faces(vec2{3.5, 0.5}).size() == 0);
  //
  //REQUIRE(ot.nearby_faces(vec2{0.5, 1.5}).size() == 1);
  //REQUIRE(ot.nearby_faces(vec2{1.5, 1.5}).size() == 1);
  //REQUIRE(ot.nearby_faces(vec2{2.5, 1.5}).size() == 0);
  //REQUIRE(ot.nearby_faces(vec2{3.5, 1.5}).size() == 0);
  //
  //REQUIRE(ot.nearby_faces(vec2{0.5, 2.5}).size() == 0);
  //REQUIRE(ot.nearby_faces(vec2{1.5, 2.5}).size() == 0);
  //REQUIRE(ot.nearby_faces(vec2{2.5, 2.5}).size() == 0);
  //REQUIRE(ot.nearby_faces(vec2{3.5, 2.5}).size() == 0);
  //
  //REQUIRE(ot.nearby_faces(vec2{0.5, 3.5}).size() == 0);
  //REQUIRE(ot.nearby_faces(vec2{1.5, 3.5}).size() == 0);
  //REQUIRE(ot.nearby_faces(vec2{2.5, 3.5}).size() == 0);
  //REQUIRE(ot.nearby_faces(vec2{3.5, 3.5}).size() == 0);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
