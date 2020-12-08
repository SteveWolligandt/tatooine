#include <tatooine/pointset.h>
#include <catch2/catch.hpp>
#include <tatooine/grid.h>
#include <tatooine/random.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD((pointset<real_t, 3>), "pointset", "[pointset]") {
  auto& prop1 = add_vertex_property<real_t>("prop1", 0);
  auto  v0    = insert_vertex(1, 2, 3);
  auto  v1    = insert_vertex(2, 4, 6);
  prop1[v0]   = 123;
  prop1[v1]   = 246;
  for (size_t i = 0; i < 8; ++i) { insert_vertex(1, 2, 3); }
  auto&                  prop2 = add_vertex_property<real_t>("prop2", 2);
  [[maybe_unused]] auto& prop3 =
      add_vertex_property<vec<real_t, 3>>("prop3", {1, 0, 2});
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
  pointset<real_t, 2> ps;
  std::vector         v{ps.insert_vertex(1, 2), ps.insert_vertex(2, 3),
                ps.insert_vertex(3, 4)};

  auto& foo = ps.add_vertex_property<int>("foo");

  foo[v[0]] = 1;
  foo[v[1]] = 2;
  foo[v[2]] = 4;

  pointset<real_t, 2> copy{ps};
  const auto&         foo_copy = copy.vertex_property<int>("foo");
  for (auto v : ps.vertices()) REQUIRE(foo[v] == foo_copy[v]);
}
//==============================================================================
TEST_CASE_METHOD((pointset<real_t, 3>), "[pointset] delete vertex",
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
TEST_CASE_METHOD((pointset<real_t, 3>), "pointset_kd_tree",
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
TEST_CASE_METHOD((pointset<real_t, 2>), "pointset_inverse_distance_weighting_sampler",
                 "[pointset][inverse_distance_weighting_sampler]") {
  random_uniform rand{-1.0, 1.0, std::mt19937_64{1234}};
  auto&          prop = add_vertex_property<real_t>("prop");
  for (size_t i = 0; i < 1000; ++i) {
    auto v = insert_vertex(rand(), rand());
    prop[v] = rand() * 10;
  }
  auto sampler = inverse_distance_weighting_sampler(prop, 0.1);
  uniform_grid<real_t, 2> gr{linspace{-1.0, 1.0, 500},
                             linspace{-1.0, 1.0, 500}};
  gr.sample_to_vertex_property(sampler, "interpolated_data");
  gr.write_vtk("inverse_distance_weighting_sampler.vtk");
}
//==============================================================================
TEST_CASE_METHOD((pointset<real_t, 2>), "pointset_moving_least_squares_sampler_2",
                 "[pointset][moving_least_squares_sampler][2d][2D]") {
  random_uniform rand{-1.0, 1.0, std::mt19937_64{1234}};
  SECTION("scalar property") {
    auto& prop = add_vertex_property<real_t>("prop");
    for (size_t i = 0; i < 1000; ++i) {
      auto v  = insert_vertex(rand(), rand());
      prop[v] = rand() * 10;
    }

    auto sampler = moving_least_squares_sampler(prop, 0.1);
    grid gr{linspace{-1.0, 1.0, 500}, linspace{-1.0, 1.0, 500}};
    gr.sample_to_vertex_property(sampler, "interpolated_data");
    gr.write_vtk("moving_least_squares_sampler_2d_scalar.vtk");
    for (auto v : vertices()) {
      CHECK(sampler(at(v)) == Approx(prop[v]));
    }
  }
  SECTION("vector property") {
    auto& prop = add_vertex_property<vec3>("prop");
    for (size_t i = 0; i < 1000; ++i) {
      auto v  = insert_vertex(rand(), rand());
      prop[v](0) = rand() * 10;
      prop[v](1) = rand() * 10;
      prop[v](2) = rand() * 10;
    }

    auto sampler = moving_least_squares_sampler(prop, 0.1);
    uniform_grid<double, 2> gr{linspace{-1.0, 1.0, 500},
                               linspace{-1.0, 1.0, 500}};
    gr.sample_to_vertex_property(sampler, "interpolated_data");
    gr.write_vtk("moving_least_squares_sampler_2d_vector.vtk");
    for (auto v : vertices()) {
      CHECK(approx_equal(sampler(at(v)), prop[v]));
    }
  }
}
//==============================================================================
TEST_CASE_METHOD((pointset<real_t, 3>), "pointset_moving_least_squares_sampler_3",
                 "[pointset][moving_least_squares_sampler][3d][3D]") {
  random_uniform rand{-1.0, 1.0, std::mt19937_64{1234}};
  SECTION("scalar property") {
    auto& prop = add_vertex_property<real_t>("prop");
    for (size_t i = 0; i < 1000; ++i) {
      auto v  = insert_vertex(rand(), rand(), rand());
      prop[v] = rand() * 10;
    }

    auto sampler = moving_least_squares_sampler(prop, 0.1);
    grid gr{linspace{-1.0, 1.0, 500}, linspace{-1.0, 1.0, 500},
            linspace{-1.0, 1.0, 500}};
    gr.sample_to_vertex_property(sampler, "interpolated_data");
    gr.write_vtk("moving_least_squares_sampler_3d.vtk");
    for (auto v : vertices()) {
      CHECK(sampler(at(v)) == Approx(prop[v]));
    }
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
