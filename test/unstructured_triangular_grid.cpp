#include <tatooine/unstructured_triangular_grid.h>
#include <tatooine/for_loop.h>
#include <tatooine/dynamic_multidim_array.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid3,
                 "unstructured_triangular_grid_vertex_add") {
  auto v0 = insert_vertex(0, 0, 0);
  auto v1 = insert_vertex(1, 0, 0);
  auto v2 = insert_vertex(0, 1, 0);
  insert_simplex(v0, v1, v2);
  REQUIRE(at(v0)(0) == 0);
  REQUIRE(at(v0)(1) == 0);
  REQUIRE(at(v0)(2) == 0);
  REQUIRE(at(v1)(0) == 1);
  REQUIRE(at(v1)(1) == 0);
  REQUIRE(at(v1)(2) == 0);
  REQUIRE(at(v2)(0) == 0);
  REQUIRE(at(v2)(1) == 1);
  REQUIRE(at(v2)(2) == 0);
}
//==============================================================================
TEST_CASE_METHOD(
    unstructured_triangular_grid2, "unstructured_simplicial_grid_triangular_2d",
    "[unstructured_simplicial_grid][unstructured_triangular_grid][2d]") {
  using this_type            = unstructured_triangular_grid2;
  using vertex_handle        = this_type::vertex_handle;
  auto const v1              = insert_vertex(0, 0);
  auto const v2              = insert_vertex(1, 0);
  auto const v3              = insert_vertex(0, 1);
  auto const c1              = insert_simplex(v1, v2, v3);
  auto const [v1_, v2_, v3_] = at(c1);
  REQUIRE(v1 == v1_);
  REQUIRE(v2 == v2_);
  REQUIRE(v3 == v3_);
  REQUIRE(typeid(v1_) == typeid(vertex_handle&));
  REQUIRE(typeid(v2_) == typeid(vertex_handle&));
  REQUIRE(typeid(v3_) == typeid(vertex_handle&));
  auto const [cv1_, cv2_, cv3_] = static_cast<this_type const&>(*this)[c1];
  REQUIRE(v1 == cv1_);
  REQUIRE(v2 == cv2_);
  REQUIRE(v3 == cv3_);
  REQUIRE(typeid(cv1_) == typeid(vertex_handle const&));
  REQUIRE(typeid(cv2_) == typeid(vertex_handle const&));
  REQUIRE(typeid(cv3_) == typeid(vertex_handle const&));
}
//==============================================================================
TEST_CASE_METHOD(
    unstructured_triangular_grid3, "unstructured_triangular_grid_remove_vertex",
    "[unstructured_triangular_grid][triangular_grid][remove][vertex]") {
  using namespace std::ranges;
  [[maybe_unused]] auto v0 = insert_vertex(0, 0, 0);
  [[maybe_unused]] auto v1 = insert_vertex(1, 0, 0);
  [[maybe_unused]] auto v2 = insert_vertex(1, 1, 0);
  [[maybe_unused]] auto v3 = insert_vertex(0, 1, 0);
  [[maybe_unused]] auto c0 = insert_simplex(v0, v1, v2);
  [[maybe_unused]] auto c1 = insert_simplex(v0, v2, v3);

  remove(v1);

  REQUIRE(find(vertices(), v1) == end(vertices()));
  REQUIRE(find(simplices(), c0) == end(simplices()));
}
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid2,
                 "unstructured_triangular_grid_simplex_contains_vertex",
                 "[unstructured_triangular_grid][triangular_grid][contains]["
                 "vertex_in_simplex]") {
  using namespace std::ranges;
  [[maybe_unused]] auto v0 = insert_vertex(0, 0);
  [[maybe_unused]] auto v1 = insert_vertex(1, 0);
  [[maybe_unused]] auto v2 = insert_vertex(1, 1);
  [[maybe_unused]] auto v3 = insert_vertex(0, 1);
  [[maybe_unused]] auto c0 = insert_simplex(v0, v1, v2);
  [[maybe_unused]] auto c1 = insert_simplex(v0, v2, v3);

  REQUIRE(contains(c0, v0));
  REQUIRE(contains(c0, v1));
  REQUIRE(contains(c0, v2));
  REQUIRE_FALSE(contains(c0, v3));
  REQUIRE(contains(c1, v0));
  REQUIRE_FALSE(contains(c1, v1));
  REQUIRE(contains(c1, v2));
  REQUIRE(contains(c1, v3));
}
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid2,
                 "unstructured_triangular_grid_tidy_up",
                 "[unstructured_triangular_grid][triangular_grid][tidyup][tidy]") {
  using this_type            = unstructured_triangular_grid2;
  using vertex_handle        = this_type::vertex_handle;
  using simplex_handle        = this_type::simplex_handle;
  using namespace std::ranges;
  auto vertex_arr    = dynamic_multidim_array<vertex_handle>{5, 5};
  auto tri_left_arr  = dynamic_multidim_array<simplex_handle>{4, 4};
  auto tri_right_arr = dynamic_multidim_array<simplex_handle>{4, 4};

  for_loop(
      [&](auto const... is) {
        vertex_arr(is...) = insert_vertex(static_cast<real_number>(is)...);
      },
      5, 5);
  for_loop(
      [&](auto const ix, auto const iy) {
        tri_left_arr(ix, iy) =
            insert_simplex(vertex_arr(ix, iy), vertex_arr(ix + 1, iy + 1),
                           vertex_arr(ix, iy + 1));
        tri_right_arr(ix, iy) =
            insert_simplex(vertex_arr(ix, iy), vertex_arr(ix + 1, iy),
                           vertex_arr(ix + 1, iy + 1));
      },
      4, 4);

  REQUIRE(size(vertices()) == 25);
  REQUIRE(size(simplices()) == 32);
  REQUIRE(size(vertex_position_data()) == 25);
  REQUIRE(size(simplex_index_data()) == 32 * 3);

  {
    INFO("Triangle (1,1), (2,1), (2,2) before remove");
    auto const s            = simplex_handle{11};
    auto const [v0, v1, v2] = at(s);
    CAPTURE(v0, v1, v2);
    REQUIRE(contains(s, vertex_arr(1, 1)));
    REQUIRE(contains(s, vertex_arr(2, 1)));
    REQUIRE(contains(s, vertex_arr(2, 2)));
  }

  remove(vertex_arr(1, 2));
  remove(vertex_arr(3, 2));
  {
    INFO("Triangle (1,1), (2,1), (2,2) after remove");
    auto const s            = simplex_handle{11};
    auto const [v0, v1, v2] = at(s);
    CAPTURE(v0, v1, v2);
    REQUIRE(contains(s, vertex_arr(1, 1)));
    REQUIRE(contains(s, vertex_arr(2, 1)));
    REQUIRE(contains(s, vertex_arr(2, 2)));
  }
  REQUIRE(vertex_arr(2,2) == vertex_handle{12});
  REQUIRE(vertex_arr(4,2) == vertex_handle{14});
  REQUIRE(at(vertex_arr(2,2))(0) == 2);
  REQUIRE(at(vertex_arr(2,2))(1) == 2);
  REQUIRE(at(vertex_arr(4,2))(0) == 4);
  REQUIRE(at(vertex_arr(4,2))(1) == 2);
  REQUIRE(size(vertices()) == 23);
  REQUIRE(size(simplices()) == 20);
  REQUIRE(size(vertex_position_data()) == 25);
  REQUIRE(size(simplex_index_data()) == 32 * 3);
  tidy_up();
  REQUIRE(at(vertex_handle{11})(0) == 2);
  REQUIRE(at(vertex_handle{11})(1) == 2);
  REQUIRE(at(vertex_handle{12})(0) == 4);
  REQUIRE(at(vertex_handle{12})(1) == 2);
  REQUIRE(size(vertices()) == 23);
  REQUIRE(size(simplices()) == 20);
  REQUIRE(size(vertex_position_data()) == 23);
  REQUIRE(size(simplex_index_data()) == 20 * 3);

  {
    INFO("Triangle (1,1), (2,1), (2,2) after tidy");
    auto const s            = simplex_handle{8};
    auto const [v0, v1, v2] = at(s);
    CAPTURE(v0, v1, v2);
    REQUIRE(contains(s, vertex_handle{6}));
    REQUIRE(contains(s, vertex_handle{7}));
    REQUIRE(contains(s, vertex_handle{11}));
  }
  
}
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid3,
                 "unstructured_triangular_grid_io",
                 "[unstructured_triangular_grid][triangular_grid][io][IO]") {
  auto v0 = insert_vertex(0, 0, 0);
  auto v1 = insert_vertex(1, 0, 0);
  auto v2 = insert_vertex(0, 1, 0);
  insert_simplex(v0, v1, v2);
  write_vtp("triangle_poly.vtp");
}
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid3,
                 "unstructured_triangular_grid_copy",
                 "[unstructured_triangular_grid][copy]") {
  auto const v0 = insert_vertex(0.0, 0.0, 0.0);
  auto const v1 = insert_vertex(1.0, 0.0, 0.0);
  auto const v2 = insert_vertex(0.0, 1.0, 0.0);
  auto const f0 = insert_simplex(v0, v1, v2);

  auto& vertex_prop  = scalar_vertex_property("vertex_prop");
  vertex_prop[v0]    = 0;
  vertex_prop[v1]    = 1;
  vertex_prop[v2]    = 2;
  auto& simplex_prop = scalar_simplex_property("simplex_prop");
  simplex_prop[f0]   = 4;

  auto copied_mesh = *this;

  REQUIRE(at(v0) == copied_mesh[v0]);
  REQUIRE(at(v1) == copied_mesh[v1]);
  REQUIRE(at(v2) == copied_mesh[v2]);
  at(v0)(0) = 2;
  REQUIRE_FALSE(at(v0) == copied_mesh[v0]);

  {
    auto& copied_vertex_prop =
        copied_mesh.scalar_vertex_property("vertex_prop");
    REQUIRE(vertex_prop[v0] == copied_vertex_prop[v0]);
    REQUIRE(vertex_prop[v1] == copied_vertex_prop[v1]);
    REQUIRE(vertex_prop[v2] == copied_vertex_prop[v2]);

    vertex_prop[v0] = 100;
    REQUIRE_FALSE(vertex_prop[v0] == copied_vertex_prop[v0]);

    auto& copied_simplex_prop =
        copied_mesh.scalar_simplex_property("simplex_prop");
    REQUIRE(simplex_prop[f0] == copied_simplex_prop[f0]);

    simplex_prop[f0] = 10;
    REQUIRE_FALSE(simplex_prop[f0] == copied_simplex_prop[f0]);
  }

  copied_mesh = *this;
  {
    auto& copied_vertex_prop =
        copied_mesh.scalar_vertex_property("vertex_prop");
    REQUIRE(at(v0) == copied_mesh[v0]);
    REQUIRE(vertex_prop[v0] == copied_vertex_prop[v0]);

    auto& copied_simplex_prop =
        copied_mesh.scalar_simplex_property("simplex_prop");
    REQUIRE(at(f0) == copied_mesh[f0]);
    REQUIRE(simplex_prop[f0] == copied_simplex_prop[f0]);
  }
}
//==============================================================================
TEST_CASE_METHOD(unstructured_triangular_grid2,
                 "unstructured_triangular_grid_linear_sampler",
                 "[unstructured_triangular_grid][linear_sampler]") {
  auto const v0 = insert_vertex(0.0, 0.0);
  auto const v1 = insert_vertex(1.0, 0.0);
  auto const v2 = insert_vertex(0.0, 1.0);
  auto const v3 = insert_vertex(1.0, 1.0);
  insert_simplex(v0, v1, v2);
  insert_simplex(v1, v3, v2);

  auto& prop   = scalar_vertex_property("prop");
  prop[v0]     = 1;
  prop[v1]     = 2;
  prop[v2]     = 3;
  prop[v3]     = 4;
  auto sampler = this->sampler(prop);
  REQUIRE(sampler(at(v0)) == prop[v0]);
  REQUIRE(sampler(at(v1)) == prop[v1]);
  REQUIRE(sampler(at(v2)) == prop[v2]);
  REQUIRE(sampler(at(v3)) == prop[v3]);
  REQUIRE(sampler(vec2{0.5, 0.5}) == Approx(2.5));
  REQUIRE(sampler(vec2{0.0, 0.0}) == Approx(1));
  REQUIRE(sampler(vec2{1.0, 1.0}) == Approx(4));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
