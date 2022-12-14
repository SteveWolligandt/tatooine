#include <tatooine/pointset.h>
#include <tatooine/random.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/test/EqualRange.h>

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
TEST_CASE_METHOD(pointset2, "pointset_tidyup", "[pointset][tidyup][tidy]") {
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
TEST_CASE_METHOD(pointset3, "pointset", "[pointset][vertex_property]") {
  auto& prop1 = scalar_vertex_property("prop1");
  auto  v0    = insert_vertex(1, 2, 3);
  auto  v1    = insert_vertex(2, 4, 6);
  prop1[v0]   = 123;
  prop1[v1]   = 246;
  for (size_t i = 0; i < 8; ++i) {
    insert_vertex(1, 2, 3);
  }
  auto&                  prop2 = insert_scalar_vertex_property("prop2", 2);
  [[maybe_unused]] auto& prop3 =
      insert_vec3_vertex_property("prop3", vec3{1, 0, 2});

  REQUIRE(size(vertices()) == 10);
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
TEST_CASE_METHOD(pointset2, "pointset_inverse_distance_weighting_sampler",
                 "[pointset][inverse_distance_weighting_sampler]") {
  random::uniform rand{-1.0, 1.0, std::mt19937_64{1234}};
  auto&           prop = scalar_vertex_property("prop");
  for (size_t i = 0; i < 100; ++i) {
    auto v  = insert_vertex(rand(), rand());
    prop[v] = rand() * 10;
  }
  auto sampler = inverse_distance_weighting_sampler(prop, 0.1);
  auto gr      = uniform_rectilinear_grid2{linspace{-1.0, 1.0, 500},
                                      linspace{-1.0, 1.0, 500}};
  gr.sample_to_vertex_property(sampler, "interpolated_data");
}
//==============================================================================
TEST_CASE_METHOD(pointset2, "pointset_vertex_range",
                 "[pointset][range][vertex_container][iterators]") {
  using vh = vertex_handle;
  auto  rand = random::uniform{-1.0, 1.0, std::mt19937_64{1234}};
  auto& prop = vertex_property<std::size_t>("prop");
  auto num_vertices = std::size_t(4);
  for (size_t i = 0; i < num_vertices; ++i) {
    insert_vertex(rand(), rand());
  }
  auto const vertices_expected = std::vector{vh{0}, vh{1}, vh{2}, vh{3}};

  SECTION("iterator") {
      auto it0           = begin(vertices());
      auto it1           = begin(vertices_expected);

      REQUIRE(it0 - next(it0) == it1 - next(it1));
      REQUIRE(next(it0) - it0 == next(it1) - it1);
      REQUIRE(distance(end(vertices()), begin(vertices())) ==
              distance(end(vertices_expected), begin(vertices_expected)));
      REQUIRE(end(vertices()) - begin(vertices()) ==
              end(vertices_expected) - begin(vertices_expected));
      REQUIRE(distance(begin(vertices()), end(vertices())) ==
              distance(begin(vertices_expected), end(vertices_expected)));
      REQUIRE(begin(vertices()) - end(vertices()) ==
              begin(vertices_expected) - end(vertices_expected));
      REQUIRE(distance(it0, next(it0)) == distance(it1, next(it1)));
  }

  SECTION("range-based for-loop"){
    SECTION("sequential") {
      for (auto const v : vertices()) {
        prop[v] = v.index();
      }
      for (auto const v : vertices()) {
        CAPTURE(v);
        REQUIRE(prop[v] == v.index());
      }
    }
    SECTION("parallel") {
#pragma omp parallel for
      for (auto const v : vertices()) {
        prop[v] = v.index();
      }
      for (auto const v : vertices()) {
        CAPTURE(v);
        REQUIRE(prop[v] == v.index());
      }
    }
  }

  SECTION("C++20 ranges library") {
    SECTION("copy") {
      auto       vs_vec = std::vector<vh>{};
      std::ranges::copy(vertices(), std::back_inserter(vs_vec));
      REQUIRE_THAT(vs_vec, EqualRange(vertices_expected));
    }
  }
}
//==============================================================================
TEST_CASE("pointset_io", "[pointset][io]") {
  SECTION("vtp"){SECTION("3D") {
    auto ps = pointset3{};
    auto v1 = ps.insert_vertex(1, 2, 3);
    auto v2 = ps.insert_vertex(2, 3, 4);
    auto const prop_vec3_name = "vec3";
    auto& prop_vec3 = ps.vec3_vertex_property(prop_vec3_name);
    prop_vec3[v1] = vec3{0.5, 1.0, 1.5};
    prop_vec3[v2] = vec3{0.6, 1.1, 1.6};
    auto const prop_scalar_name = "scalar";
    auto& prop_scalar = ps.scalar_vertex_property(prop_scalar_name);
    prop_scalar[v1] = 1;
    prop_scalar[v2] = 1.1;

    auto filepath = filesystem::path{"pointset_io_test_3d.vtp"};
    ps.write_vtp(filepath);

    auto ps_read = pointset3{filepath};
    // check if vertices are equal
    REQUIRE(ps_read.num_vertices() == ps.num_vertices());
    REQUIRE_THAT(ps[v1], EqualRange(ps_read[v1]));
    REQUIRE_THAT(ps[v2], EqualRange(ps_read[v2]));

    // check if vertex properties match
    REQUIRE(ps_read.num_vertex_properties() == ps.num_vertex_properties());
    REQUIRE(ps_read.has_vertex_property(prop_vec3_name));
    REQUIRE(ps_read.has_vertex_property(prop_scalar_name));
    REQUIRE(ps_read.vertex_properties()[prop_vec3_name]->type() == typeid(vec3));
    REQUIRE(ps_read.vertex_properties()[prop_scalar_name]->type() == typeid(real_number));

    // check if each single vertex property is equal
    auto const& prop_read_vec3 = ps_read.vec3_vertex_property(prop_vec3_name);
    REQUIRE_THAT(prop_read_vec3[v1], EqualRange(prop_read_vec3[v1]));
    REQUIRE_THAT(prop_read_vec3[v2], EqualRange(prop_read_vec3[v2]));

    auto const& prop_read_scalar = ps_read.scalar_vertex_property(prop_scalar_name);
    REQUIRE(prop_read_scalar[v1] == prop_read_scalar[v1]);
    REQUIRE(prop_read_scalar[v2] == prop_read_scalar[v2]);
  }}
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
