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
TEST_CASE("grid_face_center", "[grid][face][position]") {
  std::array  dim0{0.0, 1.0, 2.0};
  std::vector dim1{0.0, 1.0, 2.0};
  linspace    dim2{0.0, 1.0, 11};
  grid        g{dim0, dim1, dim2};
  auto const  f000 = g.face_center_at<0>(0, 0, 0);
  CAPTURE(f000);
  REQUIRE(approx_equal(f000, vec{0.0, 0.5, 0.05}));
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
  //linspace dim0{0.0, 1.0, 11};
  //linspace dim1{0.0, 1.0, 11};
  //grid     g{dim0, dim1};
  //using grid_t = decltype(g);
  //using prop_value_type = double;
  //using interpolation_kernel_t =
  //    grid_t::default_interpolation_kernel_t<prop_value_type>;
  //
  //std::string const prop_name = "u";
  //auto&             u_prop = dynamic_cast<grid_t::contiguous_vertex_property_t<
  //    prop_value_type, interpolation_kernel_t, interpolation_kernel_t>&>(
  //    g.add_vertex_property<prop_value_type>(prop_name));
  //u_prop.data_at(1, 1) = 2;
  //REQUIRE(u_prop[1][1] == 2);
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
      g.add_vertex_property<prop_value_type, interpolation::linear,
                                    interpolation::linear>(prop_name);
  u_prop.data_at(4, 1) = 5;
  u_prop.data_at(5, 1) = 6;
  u_prop.data_at(4, 2) = 7;
  u_prop.data_at(5, 2) = 8;
  REQUIRE(u_prop.sample(0.41, 0.11) == Approx(5.3));
}
//==============================================================================
TEST_CASE("grid_finite_differences_coefficients", "[grid][finite_differences]") {
  std::array dim{0.0, 0.1, 0.5, 0.9, 1.0};
  grid     g{dim};
  using prop_value_type = double;

  std::string const prop_name = "u";
  auto&             u_prop =
      g.add_vertex_property<prop_value_type>(prop_name);
  auto [i0, coeffs0]= u_prop.stencil_coefficients<0, 3>(0, 1);
  REQUIRE(i0 == 0);
  REQUIRE(coeffs0(0) == Approx(-12.0));
  REQUIRE(coeffs0(1) == Approx(12.5));
  REQUIRE(coeffs0(2) == Approx(-0.5));
  auto [i1, coeffs1] = u_prop.stencil_coefficients<0, 3>(1, 1);
  REQUIRE(i1 == 0);
  REQUIRE(coeffs1(0) == Approx(-8.0));
  REQUIRE(coeffs1(1) == Approx(7.5));
  REQUIRE(coeffs1(2) == Approx(0.5));
  auto [i2, coeffs2] = u_prop.stencil_coefficients<0, 3>(2, 1);
  REQUIRE(i2 == 1);
  REQUIRE(coeffs2(0) == Approx(-5.0/4));
  REQUIRE(coeffs2(1) == Approx(0));
  REQUIRE(coeffs2(2) == Approx(5.0/4));
  auto [i3, coeffs3] = u_prop.stencil_coefficients<0, 3>(3, 1);
  REQUIRE(i3 == 2);
  REQUIRE(coeffs3(0) == Approx(-0.5));
  REQUIRE(coeffs3(1) == Approx(-7.5));
  REQUIRE(coeffs3(2) == Approx(8));
  auto [i4, coeffs4] = u_prop.stencil_coefficients<0, 3>(4, 1);
  REQUIRE(i4 == 2);
  REQUIRE(coeffs4(0) == Approx(0.5));
  REQUIRE(coeffs4(1) == Approx(-12.5));
  REQUIRE(coeffs4(2) == Approx(12));
}
//==============================================================================
//TEST_CASE("grid_diff", "[grid][sample][1d][hermite]") {
//  std::array dim{0.0, 0.1, 0.5, 0.9, 1.0};
//  grid       g{dim};
//  using prop_value_type = double;
//
//  std::string const prop_name = "u";
//  auto&             u_prop =
//      g.add_vertex_property<prop_value_type, interpolation::hermite>(prop_name);
//  u_prop.data_at(0) = 0;
//  u_prop.data_at(1) = 1;
//  u_prop.data_at(2) = 2;
//  u_prop.data_at(3) = 3;
//  u_prop.data_at(4) = 4;
//  for (auto const t : linspace{0.0, 1.0, 21}) {
//    std::cerr << u_prop.sample(t) << ' ';
//  }
//  std::cerr << '\n';
//}
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
TEST_CASE("grid_face_property", "[grid][face_property]") {
  std::array  dim0{-1.0, 1.0, 2.0, 5.0, 6.0};
  std::vector dim1{-3.0, 0.0, 2.0, 3.0, 4.0};
  grid        g{dim0, dim1};

  auto& u_prop = g.add_face_property<double, 0>("u");
  auto& v_prop = g.add_chunked_face_property<double, 1, 128>("v");
  u_prop.data_at(0, 0) = 3;
  u_prop.data_at(1, 0) = 5;
  u_prop.data_at(0, 1) = 6;
  u_prop.data_at(1, 1) = 8;
  SECTION("on data points") {
    REQUIRE(u_prop.sample(-1, -1.5) == Approx(3));
    REQUIRE(u_prop.sample(1, -1.5) == Approx(5));
    REQUIRE(u_prop.sample(-1, 1) == Approx(6));
    REQUIRE(u_prop.sample(1, 1) == Approx(8));
  }

  SECTION("on x-edge") {
    REQUIRE(u_prop.sample(0, -1.5) == Approx(4));
    REQUIRE(u_prop.sample(0, 1) == Approx(7));
  }

  SECTION("on y-edge") {
    REQUIRE(u_prop.sample(-1, -0.25) == Approx(4.5));
    REQUIRE(u_prop.sample(1, -0.25) == Approx(6.5));
  }

  SECTION("center") {
    REQUIRE(u_prop.sample(0, -0.25) == Approx(5.5));
  }

  v_prop.data_at(0, 0) = 4;
  v_prop.data_at(1, 0) = 6;

}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
