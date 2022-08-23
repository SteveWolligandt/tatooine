#include <tatooine/pointset.h>
#include <tatooine/random.h>
#include <tatooine/rectilinear_grid.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(pointset2, "pointset_vtp_matrix",
                 "[pointset][write][vtp][matrix]") {
  auto  v0       = insert_vertex(1, 2);
  auto  v1       = insert_vertex(2, 4);
  auto& mat_prop = mat2_vertex_property("mat2");
  mat_prop[v0]   = mat2{{1, 2}, {3, 4}};
  mat_prop[v1]   = mat2{{5, 6}, {7, 8}};
  write_vtp("pointset.unittests.vtp_matrix.vtp");
}
//==============================================================================
TEST_CASE_METHOD(pointset3, "pointset_ranges", "[pointset][ranges]") {
  auto                  v0    = insert_vertex(1, 2, 3);
  [[maybe_unused]] auto v1    = insert_vertex(2, 4, 6);
  auto                  is0   = std::vector<std::size_t>{};
  constexpr auto        index = [](auto const vh) { return vh.index(); };
  using namespace std::ranges;
  copy(vertices() | views::transform(index), std::back_inserter(is0));
  remove(v0);
  auto is1       = std::vector<std::size_t>{};
  auto get_index = [](auto const vh) { return vh.index(); };
  copy(vertices() | views::transform(get_index), std::back_inserter(is1));

  REQUIRE(size(is0) == 2);
  REQUIRE(is0[0] == 0);
  REQUIRE(is0[1] == 1);
  REQUIRE(size(is1) == 1);
  REQUIRE(is1[0] == 1);
}
//==============================================================================
TEST_CASE_METHOD(pointset2, "pointset_tidyup", "[pointset][tidyup]") {
  [[maybe_unused]] auto v0 = insert_vertex(1, 2);
  [[maybe_unused]] auto v1 = insert_vertex(2, 4);
  [[maybe_unused]] auto v2 = insert_vertex(3, 6);
  [[maybe_unused]] auto v3 = insert_vertex(4, 8);

  REQUIRE(size(vertices()) == 4);
  REQUIRE(size(vertex_position_data()) == 4);
  REQUIRE(vertex_at(0)(0) == 1);
  REQUIRE(vertex_at(0)(1) == 2);
  REQUIRE(vertex_at(1)(0) == 2);
  REQUIRE(vertex_at(1)(1) == 4);
  REQUIRE(vertex_at(2)(0) == 3);
  REQUIRE(vertex_at(2)(1) == 6);
  REQUIRE(vertex_at(3)(0) == 4);
  REQUIRE(vertex_at(3)(1) == 8);

  remove(v1);
  REQUIRE(size(vertices()) == 3);
  REQUIRE(size(vertex_position_data()) == 4);
  REQUIRE(vertex_at(0)(0) == 1);
  REQUIRE(vertex_at(0)(1) == 2);
  REQUIRE(vertex_at(1)(0) == 2);
  REQUIRE(vertex_at(1)(1) == 4);
  REQUIRE(vertex_at(2)(0) == 3);
  REQUIRE(vertex_at(2)(1) == 6);
  REQUIRE(vertex_at(3)(0) == 4);
  REQUIRE(vertex_at(3)(1) == 8);

  tidy_up();
  REQUIRE(size(vertices()) == 3);
  REQUIRE(size(vertex_position_data()) == 3);
  REQUIRE(vertex_at(0)(0) == 1);
  REQUIRE(vertex_at(0)(1) == 2);
  REQUIRE(vertex_at(1)(0) == 3);
  REQUIRE(vertex_at(1)(1) == 6);
  REQUIRE(vertex_at(2)(0) == 4);
  REQUIRE(vertex_at(2)(1) == 8);
}
//==============================================================================
TEST_CASE_METHOD(pointset3, "pointset", "[pointset][general]") {
  auto& prop1 = insert_scalar_vertex_property("prop1", 0);
  auto  v0    = insert_vertex(1, 2, 3);
  auto  v1    = insert_vertex(2, 4, 6);
  prop1[v0]   = 123;
  prop1[v1]   = 246;
  for (size_t i = 0; i < 8; ++i) {
    insert_vertex(1, 2, 3);
  }
  auto&                  prop2 = insert_scalar_vertex_property("prop2", 2);
  [[maybe_unused]] auto& prop3 =
      insert_vec3_vertex_property("prop3", {1, 0, 2});

  REQUIRE(vertices().size() == 10);
  REQUIRE(prop1.size() == 10);
  REQUIRE(prop1[v0] == 123);
  REQUIRE(prop1[v1] == 246);

  REQUIRE(prop2.size() == 10);
  for (auto v : vertices()) {
    REQUIRE(prop2[v] == 2);
  }

  remove(v1);
  tidy_up();

  REQUIRE(vertices().size() == 9);
  REQUIRE(prop1.size() == 9);
  REQUIRE(prop2.size() == 9);
  REQUIRE(prop1[v0] == 123);
  REQUIRE(prop1[v1] == 0);
  for (auto v : vertices()) {
    REQUIRE(prop2[v] == 2);
  }
}
//==============================================================================
TEST_CASE("pointset_copy", "[pointset][copy]") {
  pointset2   ps;
  std::vector v{ps.insert_vertex(1, 2), ps.insert_vertex(2, 3),
                ps.insert_vertex(3, 4)};

  auto& foo = ps.vertex_property<int>("foo");

  foo[v[0]] = 1;
  foo[v[1]] = 2;
  foo[v[2]] = 4;

  pointset2   copy{ps};
  const auto& foo_copy = copy.vertex_property<int>("foo");
  for (auto v : ps.vertices()) {
    REQUIRE(foo[v] == foo_copy[v]);
  }
}
//==============================================================================
TEST_CASE_METHOD(pointset3, "pointset_delete_vertex",
                 "[pointset][delete][vertex]") {
  auto v0 = insert_vertex(1, 2, 3);
  insert_vertex(2, 3, 4);
  insert_vertex(3, 4, 5);
  insert_vertex(4, 5, 6);
  REQUIRE(vertices().size() == 4);
  REQUIRE(vertices().data_container().size() == 4);
  remove(v0);
  REQUIRE(vertices().size() == 3);
  REQUIRE(vertices().data_container().size() == 4);
  tidy_up();
  REQUIRE(vertices().size() == 3);
  REQUIRE(vertices().data_container().size() == 3);
  SECTION("v0 now must be {2,3,4}") { REQUIRE(at(v0)(0) == 2); }
}
//==============================================================================
// TEST_CASE_METHOD(pointset3, "pointset_kd_tree", "[pointset][kdtree]") {
//  auto const v0 = insert_vertex(0, 0, 0);
//  auto const v1 = insert_vertex(1, 0, 0);
//  auto const v2 = insert_vertex(-1, 0, 0);
//
//  REQUIRE(nearest_neighbor(vec3{0.1, 0, 0}) == v0);
//  REQUIRE(nearest_neighbor(vec3{0.49999, 0, 0}) == v0);
//  REQUIRE(nearest_neighbor(vec3{0.500001, 0, 0}) == v1);
//  REQUIRE(nearest_neighbor(vec3{-0.500001, 0, 0}) == v2);
//  auto const nearest_0_5_2 = nearest_neighbors(vec3{0.5, 0, 0}, 2);
//  REQUIRE(size(nearest_0_5_2) == 2);
//  REQUIRE((nearest_0_5_2[0] == v0 || nearest_0_5_2[1] == v0));
//  REQUIRE((nearest_0_5_2[0] == v1 || nearest_0_5_2[1] == v1));
//}
//==============================================================================
TEST_CASE_METHOD((pointset2), "pointset_inverse_distance_weighting_sampler",
                 "[pointset][inverse_distance_weighting_sampler]") {
  random::uniform rand{-1.0, 1.0, std::mt19937_64{1234}};
  auto&           prop = scalar_vertex_property("prop");
  for (size_t i = 0; i < 100; ++i) {
    auto v  = insert_vertex(rand(), rand());
    prop[v] = rand() * 10;
  }
  auto sampler = inverse_distance_weighting_sampler(prop, 0.1);
  uniform_rectilinear_grid2 gr{linspace{-1.0, 1.0, 500},
                               linspace{-1.0, 1.0, 500}};
  gr.sample_to_vertex_property(sampler, "interpolated_data");
  gr.write_vtk("inverse_distance_weighting_sampler.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
