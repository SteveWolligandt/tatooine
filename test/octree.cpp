#include <tatooine/triangular_mesh.h>
#include <tatooine/uniform_tree_hierarchy.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("uniform_tree_hierarchy", "[uniform_tree_hierarchy]") {
  triangular_mesh3 mesh;
  auto             v0 = mesh.insert_vertex(0.5, 0.5, 0.5);
  uniform_tree_hierarchy<triangular_mesh3> hierarchy{mesh, vec3::zeros(),
                                                     vec3::ones() * 4, 2};

  REQUIRE_FALSE(hierarchy.is_splitted());
  REQUIRE_FALSE(hierarchy.holds_vertices());

  hierarchy.insert_vertex(v0);
  REQUIRE_FALSE(hierarchy.is_splitted());
  REQUIRE(hierarchy.holds_vertices());

  hierarchy.insert_vertex(mesh.insert_vertex(1.5, 0.5, 0.5));
  REQUIRE(hierarchy.is_splitted());
  REQUIRE_FALSE(hierarchy.holds_vertices());
  REQUIRE_FALSE(hierarchy.child_at(0, 0, 0)->holds_vertices());
  REQUIRE_FALSE(hierarchy.child_at(1, 0, 0)->holds_vertices());
  REQUIRE_FALSE(hierarchy.child_at(0, 1, 0)->holds_vertices());
  REQUIRE_FALSE(hierarchy.child_at(1, 1, 0)->holds_vertices());
  REQUIRE_FALSE(hierarchy.child_at(0, 0, 1)->holds_vertices());
  REQUIRE_FALSE(hierarchy.child_at(1, 0, 1)->holds_vertices());
  REQUIRE_FALSE(hierarchy.child_at(0, 1, 1)->holds_vertices());
  REQUIRE_FALSE(hierarchy.child_at(1, 1, 1)->holds_vertices());

  REQUIRE(
      hierarchy.child_at(0, 0, 0)->child_at(0, 0, 0)->num_vertex_handles() ==
      1);
  REQUIRE(
      hierarchy.child_at(0, 0, 0)->child_at(1, 0, 0)->num_vertex_handles() ==
      1);

  hierarchy.insert_vertex(mesh.insert_vertex(0.75, 0.5, 0.5));
  REQUIRE(
      hierarchy.child_at(0, 0, 0)->child_at(0, 0, 0)->num_vertex_handles() ==
      2);

  hierarchy.insert_vertex(mesh.insert_vertex(2, 2, 2));
  REQUIRE(
      hierarchy.child_at(0, 0, 0)->child_at(1, 1, 1)->num_vertex_handles() ==
      1);
  REQUIRE(hierarchy.child_at(1, 0, 0)->num_vertex_handles() == 1);
  REQUIRE(hierarchy.child_at(0, 1, 0)->num_vertex_handles() == 1);
  REQUIRE(hierarchy.child_at(1, 1, 0)->num_vertex_handles() == 1);
  REQUIRE(hierarchy.child_at(0, 0, 1)->num_vertex_handles() == 1);
  REQUIRE(hierarchy.child_at(1, 0, 1)->num_vertex_handles() == 1);
  REQUIRE(hierarchy.child_at(0, 1, 1)->num_vertex_handles() == 1);
  REQUIRE(hierarchy.child_at(1, 1, 1)->num_vertex_handles() == 1);

  // auto v1 = mesh.insert_vertex(1.75, 0.5);
  // hierarchy.insert_vertex(mesh, v1.i);
  // REQUIRE(hierarchy.left_bottom()->left_bottom()->num_vertex_handles() == 2);
  // REQUIRE(hierarchy.left_bottom()->right_bottom()->num_vertex_handles() ==
  // 2); REQUIRE(hierarchy.left_bottom()->left_top()->num_vertex_handles() ==
  // 0); REQUIRE(hierarchy.left_bottom()->right_top()->num_vertex_handles() ==
  // 1);
  //
  // auto v2 = mesh.insert_vertex(0.5, 1.75);
  // hierarchy.insert_vertex(mesh, v2.i);
  // REQUIRE(hierarchy.left_bottom()->left_bottom()->num_vertex_handles() == 2);
  // REQUIRE(hierarchy.left_bottom()->right_bottom()->num_vertex_handles() ==
  // 2); REQUIRE(hierarchy.left_bottom()->left_top()->num_vertex_handles() ==
  // 1); REQUIRE(hierarchy.left_bottom()->right_top()->num_vertex_handles() ==
  // 1);
  //
  // hierarchy.insert_face(mesh, mesh.insert_face(v0, v1, v2).i);
  // REQUIRE(hierarchy.left_bottom()->left_bottom()->num_vertex_handles() == 2);
  // REQUIRE(hierarchy.left_bottom()->right_bottom()->num_vertex_handles() ==
  // 2); REQUIRE(hierarchy.left_bottom()->left_top()->num_vertex_handles() ==
  // 1); REQUIRE(hierarchy.left_bottom()->right_top()->num_vertex_handles() ==
  // 1); REQUIRE(hierarchy.left_bottom()->left_bottom()->num_face_handles() ==
  // 1); REQUIRE(hierarchy.left_bottom()->right_bottom()->num_face_handles() ==
  // 1); REQUIRE(hierarchy.left_bottom()->left_top()->num_face_handles() == 1);
  // REQUIRE(hierarchy.left_bottom()->right_top()->num_face_handles() == 1);
  // REQUIRE_FALSE(hierarchy.right_bottom()->is_splitted());
  // REQUIRE_FALSE(hierarchy.right_top()->is_splitted());
  // REQUIRE_FALSE(hierarchy.left_top()->is_splitted());
  // REQUIRE_FALSE(hierarchy.right_bottom()->holds_faces());
  // REQUIRE_FALSE(hierarchy.right_top()->holds_faces());
  // REQUIRE_FALSE(hierarchy.left_top()->holds_faces());
  //
  //// face get
  // REQUIRE(hierarchy.nearby_faces(vec2{0.5, 0.5}).size() == 1);
  // REQUIRE(hierarchy.nearby_faces(vec2{1.5, 0.5}).size() == 1);
  // REQUIRE(hierarchy.nearby_faces(vec2{2.5, 0.5}).size() == 0);
  // REQUIRE(hierarchy.nearby_faces(vec2{3.5, 0.5}).size() == 0);
  //
  // REQUIRE(hierarchy.nearby_faces(vec2{0.5, 1.5}).size() == 1);
  // REQUIRE(hierarchy.nearby_faces(vec2{1.5, 1.5}).size() == 1);
  // REQUIRE(hierarchy.nearby_faces(vec2{2.5, 1.5}).size() == 0);
  // REQUIRE(hierarchy.nearby_faces(vec2{3.5, 1.5}).size() == 0);
  //
  // REQUIRE(hierarchy.nearby_faces(vec2{0.5, 2.5}).size() == 0);
  // REQUIRE(hierarchy.nearby_faces(vec2{1.5, 2.5}).size() == 0);
  // REQUIRE(hierarchy.nearby_faces(vec2{2.5, 2.5}).size() == 0);
  // REQUIRE(hierarchy.nearby_faces(vec2{3.5, 2.5}).size() == 0);
  //
  // REQUIRE(hierarchy.nearby_faces(vec2{0.5, 3.5}).size() == 0);
  // REQUIRE(hierarchy.nearby_faces(vec2{1.5, 3.5}).size() == 0);
  // REQUIRE(hierarchy.nearby_faces(vec2{2.5, 3.5}).size() == 0);
  // REQUIRE(hierarchy.nearby_faces(vec2{3.5, 3.5}).size() == 0);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
