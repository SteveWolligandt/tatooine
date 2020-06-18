#include <tatooine/grid.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("grid_vertex_indexing", "[grid][vertex][indexing]") {
  std::array  dim0{0, 1, 2};
  std::vector dim1{0, 1, 2};
  linspace    dim2{0.0, 1.0, 11};
  grid        g{dim0, dim1, dim2};
  auto const  v000 = g.vertices().at(0, 0, 0);
  auto const  v111 = g.vertex_at(1, 1, 1);
  auto const  v221 = g.vertices().at(2, 2, 1);

  REQUIRE(approx_equal(v000, vec{0.0, 0.0, 0.0}));
  REQUIRE(approx_equal(v111, vec{1.0, 1.0, 0.1}));
  REQUIRE(approx_equal(v221, vec{2.0, 2.0, 0.1}));
}
//==============================================================================
TEST_CASE("grid_vertex_iterator", "[grid][vertex][iterator]") {
  std::array  dim0{0, 1, 2};
  std::vector dim1{0, 1, 2};
  linspace    dim2{0.0, 2.0, 3};
  grid        g{dim0, dim1, dim2};
  auto        it = begin(vertices(g));
  REQUIRE(approx_equal(*it, g(0, 0, 0)));
  REQUIRE(approx_equal(*next(it), g(1, 0, 0)));
  REQUIRE(approx_equal(*next(it, 3), g(0, 1, 0)));
  REQUIRE(approx_equal(*next(it, 9), g(0, 0, 1)));
  REQUIRE(next(it, 27) == end(vertices(g)));
}
//==============================================================================
TEST_CASE("grid_cell_index", "[grid][cell_index]") {
  std::array  dim0{0.0, 1.2, 1.3};
  std::vector dim1{0, 1, 2, 4};
  linspace    dim2{0.0, 2.0, 3};
  grid        g{dim0, dim1, dim2};
  auto const [idx0, factor0] = g.cell_index<0>(1.25);
  REQUIRE(idx0 == 1);
  REQUIRE(factor0 == 0.5);
  auto const [idx1, factor1] = g.cell_index<0>(0.1);
  REQUIRE(idx1 == 0);
  REQUIRE(factor1 == 0.1/1.2);
  auto const [idx2, factor2] = g.cell_index<1>(2.5);
  REQUIRE(idx2 == 2);
  REQUIRE(factor2 == 0.25);
  auto const [idx3, factor3] = g.cell_index<2>(1.75);
  REQUIRE(idx3 == 1);
  REQUIRE(factor3 == 0.75);
}
//==============================================================================
//TEST_CASE("grid_chunked_vertex_property", "[grid][vertex][chunked][property]") {
//  std::array  dim0{0, 1, 2};
//  std::vector dim1{0, 1, 2};
//  linspace    dim2{0.0, 2.0, 3};
//  grid        g{dim0, dim1, dim2};
//
//  auto& u_prop = g.add_chunked_vertex_property<double>("u");
//  REQUIRE(u_prop(0, 0, 0) == 0);
//  u_prop(0, 0, 0) = 1;
//  REQUIRE(u_prop(0, 0, 0) == 1);
//
//  REQUIRE(u_prop(0, 1, 2) == 0);
//  u_prop(0, 1, 2) = 3;
//  REQUIRE(u_prop(0, 1, 2) == 3);
//
//  REQUIRE_NOTHROW(g.vertex_property<double>("u"));
//  REQUIRE_THROWS(g.vertex_property<float>("u"));
//  REQUIRE_THROWS(g.vertex_property<float>("v"));
//
//  auto&       v_prop = g.add_vertex_property<float,
//                                               interpolation::linear,
//                                               interpolation::linear,
//                                               interpolation::linear>("v");
//  REQUIRE(v_prop(0, 0, 0) == 0);
//  v_prop(0, 0, 0) = 1;
//  REQUIRE(v_prop(0, 0, 0) == 1);
//
//  REQUIRE(v_prop(0, 1, 2) == 0);
//  v_prop(0, 1, 2) = 3;
//  REQUIRE(v_prop(0, 1, 2) == 3);
//
//  REQUIRE_NOTHROW(g.vertex_property<double>("u"));
//  REQUIRE_NOTHROW(g.vertex_property<float>("v"));
//  REQUIRE_THROWS(g.vertex_property<double>("v"));
//}
//==============================================================================
TEST_CASE("grid_sampler_view", "[grid][sampler][view][vertex][property]") {
  linspace dim0{0.0, 1.0, 11};
  linspace dim1{0.0, 1.0, 11};
  grid     g{dim0, dim1};
  using grid_t = decltype(g);
  using prop_value_type = double;
  using interpolation_kernel_t =
      grid_t::default_interpolation_kernel_t<prop_value_type>;

  std::string const prop_name = "u";
  auto&             u_prop    = dynamic_cast<grid_t::contiguous_sampler_t<
      prop_value_type, interpolation_kernel_t, interpolation_kernel_t>&>(
      g.add_vertex_property<prop_value_type>(prop_name));
  u_prop.data_at(1, 1) = 2;
  REQUIRE(u_prop[1][1] == 2);
}
//==============================================================================
TEST_CASE("grid_sample_1d_linear", "[grid][sampler][1d][linear]") {
  linspace dim{0.0, 1.0, 11};
  grid     g{dim};
  using prop_value_type = double;

  std::string const prop_name = "u";
  auto&             u_prop = g.add_vertex_property<prop_value_type, interpolation::linear>(prop_name);
  u_prop.data_at(4) = 5;
  u_prop.data_at(5) = 6;
  REQUIRE(u_prop.sample(0.41) == 5.1);
}
//==============================================================================
TEST_CASE("grid_sample_2d_linear", "[grid][sampler][2d][linear]") {
  linspace dim1{0.0, 1.0, 11};
  linspace dim2{0.0, 1.0, 11};
  grid     g{dim1, dim2};
  using prop_value_type = double;

  std::string const prop_name = "u";
  auto&             u_prop =
      g.add_chunked_vertex_property<prop_value_type, 16, interpolation::linear,
                                    interpolation::linear>(prop_name);
  u_prop.data_at(4, 1) = 5;
  u_prop.data_at(5, 1) = 6;
  u_prop.data_at(4, 2) = 7;
  u_prop.data_at(5, 2) = 8;
  REQUIRE(u_prop.sample(0.41, 0.11) == Approx(5.3));
}
//==============================================================================
//TEST_CASE("grid_sample_1d_hermite", "[grid][sampler][1d][hermite]") {
//  linspace dim{0.0, 1.0, 11};
//  grid     g{dim};
//  using prop_value_type = double;
//
//  std::string const prop_name = "u";
//  auto&             u_prop =
//      g.add_vertex_property<prop_value_type, interpolation::hermite>(prop_name);
//  u_prop.data_at(4) = 5;
//  u_prop.data_at(5) = 6;
//  REQUIRE(u_prop.sample(0.45) > 4);
//  REQUIRE(u_prop.sample(0.45) < 5);
//}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
