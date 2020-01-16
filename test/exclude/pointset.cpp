#include <tatooine/pointset.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE_METHOD((pointset<double, 3>), "pointset", "[pointset]") {
  auto& prop1 = add_vertex_property<double>("prop1", 0);
  auto  v0    = insert_vertex(1, 2, 3);
  auto  v1    = insert_vertex(2, 4, 6);
  prop1[v0]   = 123;
  prop1[v1]   = 246;
  for (size_t i = 0; i < 8; ++i) { insert_vertex(1, 2, 3); }
  auto&                  prop2 = add_vertex_property<double>("prop2", 2);
  [[maybe_unused]] auto& prop3 =
      add_vertex_property<vec<double, 3>>("prop3", {1, 0, 2});
  write_vtk("pointset.vtk");

  REQUIRE(num_vertices() == 10);
  REQUIRE(prop1.size() == 10);
  REQUIRE(prop1[v0] == 123);
  REQUIRE(prop1[v1] == 246);

  REQUIRE(prop2.size() == 10);
  for (auto v : vertices()) { REQUIRE(prop2[v] == 2); }

  remove(v1);
  tidy_up();

  REQUIRE(num_vertices() == 9);
  REQUIRE(prop1.size() == 9);
  REQUIRE(prop2.size() == 9);
  REQUIRE(prop1[v0] == 123);
  REQUIRE(prop1[v1] == 0);
  for (auto v : vertices()) REQUIRE(prop2[v] == 2);
}

//==============================================================================
TEST_CASE("[pointset] copy", "[pointset]") {
  pointset<double, 2> ps;
  std::vector         v{ps.insert_vertex(1, 2), ps.insert_vertex(2, 3),
                ps.insert_vertex(3, 4)};

  auto& foo = ps.add_vertex_property<int>("foo");

  foo[v[0]] = 1;
  foo[v[1]] = 2;
  foo[v[2]] = 4;

  pointset<double, 2> copy{ps};
  const auto&         foo_copy = copy.vertex_property<int>("foo");
  for (auto v : ps.vertices()) REQUIRE(foo[v] == foo_copy[v]);
}

//==============================================================================
TEST_CASE_METHOD((pointset<double, 3>), "[pointset] delete vertex",
                 "[pointset]") {
  auto v0 = insert_vertex(1, 2, 3);
  insert_vertex(2, 3, 4);
  insert_vertex(3, 4, 5);
  insert_vertex(4, 5, 6);
  REQUIRE(num_vertices() == 4);
  remove(v0);
  REQUIRE(num_vertices() == 3);
  tidy_up();
  REQUIRE(num_vertices() == 3);
  SECTION("v0 now must be {2,3,4}") { REQUIRE(at(v0)(0) == 2); }
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
