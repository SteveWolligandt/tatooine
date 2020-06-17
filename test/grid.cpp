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
TEST_CASE("grid_sample_vertex_property", "[grid][sample][vertex][property]") {
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
}  // namespace tatooine::test
//==============================================================================
