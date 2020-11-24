#include <tatooine/pointset.h>
#include <catch2/catch.hpp>
#include <tatooine/grid.h>
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
TEST_CASE_METHOD((pointset<double, 3>), "pointset_kd_tree",
                 "[pointset][kdtree]") {
  auto const v0 = insert_vertex(0, 0, 0);
  auto const v1 = insert_vertex(1, 0, 0);
  auto const v2 = insert_vertex(-1, 0, 0);

  REQUIRE(nearest_neighbor(vec3{0.1, 0, 0}) == v0);
  REQUIRE(nearest_neighbor(vec3{0.49999, 0, 0}) == v0);
  REQUIRE(nearest_neighbor(vec3{0.500001, 0, 0}) == v1);
  REQUIRE(nearest_neighbor(vec3{-0.500001, 0, 0}) == v2);
  auto const nearest_0_5_2 = nearest_neighbors(vec3{0.5, 0, 0}, 2);
  REQUIRE(size(nearest_0_5_2) == 2);
  REQUIRE((nearest_0_5_2[0] == v0 || nearest_0_5_2[1] == v0));
  REQUIRE((nearest_0_5_2[0] == v1 || nearest_0_5_2[1] == v1));
}
//==============================================================================
TEST_CASE_METHOD((pointset<double, 2>), "pointset_inverse_distance_weighting_sampler",
                 "[pointset][inverse_distance_weighting_sampler]") {
  auto const v0 = insert_vertex(-1, 0);
  auto const v1 = insert_vertex(0, 0);
  auto const v2 = insert_vertex(1, 0);

  auto& prop = add_vertex_property<double>("prop");
  prop[v0] = -1;
  prop[v1] = 0;
  prop[v2] = 1;

  auto sampler = inverse_distance_weighting_sampler(prop);
  //REQUIRE(sampler(0.5, 0, 0) == 1.5);
  uniform_grid_2d<double> gr{linspace{-2.0, 2.0, 500},
                             linspace{-2.0, 2.0, 500}};
  auto& interpolated_data = gr.add_vertex_property<double>("interpolated_data");
  gr.loop_over_vertex_indices(
      [&](auto const... is) { interpolated_data(is...) = sampler(gr(is...)); });
  gr.write_vtk("inverse_distance_weighting_sampler.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================