#include <tatooine/grid.h>
#include <tatooine/lazy_netcdf_reader.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("grid_copy_constructor", "[grid][copy][constructor]") {
  std::array                           dim0{0, 1, 2};
  std::array                           dim1{0, 1, 2};
  grid<decltype(dim0), decltype(dim1)> g0{dim0, dim1};
  auto& prop = g0.add_contiguous_vertex_property<double, x_fastest>("prop");

  prop(0, 0)      = 100;
  auto  g1        = g0;
  auto& prop_copy = g1.vertex_property<double>("prop");
  REQUIRE(prop(0, 0) == prop_copy(0, 0));
  prop(0, 0) = 0;
  REQUIRE_FALSE(prop(0, 0) == prop_copy(0, 0));
}
//==============================================================================
TEST_CASE("grid_vertex_indexing", "[grid][vertex][indexing]") {
  std::array                                           dim0{0, 1, 2};
  std::vector                                          dim1{0, 1, 2};
  linspace                                             dim2{0.0, 1.0, 11};
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};
  auto const v000 = g.vertices().at(0, 0, 0);
  auto const v111 = g.vertex_at(1, 1, 1);
  auto const v221 = g.vertices().at(2, 2, 1);

  REQUIRE(approx_equal(v000, vec{0.0, 0.0, 0.0}));
  REQUIRE(approx_equal(v111, vec{1.0, 1.0, 0.1}));
  REQUIRE(approx_equal(v221, vec{2.0, 2.0, 0.1}));
}
//==============================================================================
TEST_CASE("grid_vertex_iterator", "[grid][vertex][iterator]") {
  std::array                                           dim0{0, 1, 2};
  std::vector                                          dim1{0, 1, 2};
  linspace                                             dim2{0.0, 2.0, 3};
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};
  auto                                                 it = begin(vertices(g));
  REQUIRE(approx_equal(*it, g(0, 0, 0)));
  REQUIRE(approx_equal(*next(it), g(1, 0, 0)));
  REQUIRE(approx_equal(*next(it, 3), g(0, 1, 0)));
  REQUIRE(approx_equal(*next(it, 9), g(0, 0, 1)));
  REQUIRE(next(it, 27) == end(vertices(g)));
}
//==============================================================================
TEST_CASE("grid_cell_index", "[grid][cell_index]") {
  std::array                                           dim0{0.0, 1.2, 1.3};
  std::vector                                          dim1{0, 1, 2, 4};
  linspace                                             dim2{0.0, 2.0, 3};
  linspace                                             dim3{0.0, 1.0, 25};
  grid<decltype(dim0), decltype(dim1), decltype(dim2), decltype(dim3)> g{
      dim0, dim1, dim2, dim3};
  auto const [idx0, factor0] = g.cell_index<0>(1.25);
  REQUIRE(idx0 == 1);
  REQUIRE(factor0 == 0.5);
  auto const [idx1, factor1] = g.cell_index<0>(0.1);
  REQUIRE(idx1 == 0);
  REQUIRE(factor1 == 0.1 / 1.2);
  auto const [idx2, factor2] = g.cell_index<1>(2.5);
  REQUIRE(idx2 == 2);
  REQUIRE(factor2 == 0.25);
  auto const [idx3, factor3] = g.cell_index<2>(1.75);
  REQUIRE(idx3 == 1);
  REQUIRE(factor3 == 0.75);
  auto const [idx4, factor4] = g.cell_index<2>(2.0);
  REQUIRE(idx4 == 1);
  REQUIRE(factor4 == 1);
  auto const [idx5, factor5] = g.cell_index<0>(1.3);
  REQUIRE(idx5 == 1);
  REQUIRE(factor5 == 1);
  auto const [idx6, factor6] = g.cell_index<1>(4);
  REQUIRE(idx6 == 2);
  REQUIRE(factor6 == 1);
  auto const [idx7, factor7] = g.cell_index<3>(0.97967751460911501);
  REQUIRE(idx7 == 23);
  //REQUIRE(factor6 == 1);
}
//==============================================================================
TEST_CASE("grid_vertex_property", "[grid][property]") {
  grid g{linspace{0.0, 1.0, 11}, linspace{0.0, 1.0, 11}};
  SECTION("contiguous") {
    auto& prop = g.add_contiguous_vertex_property<double>("double_prop");
    REQUIRE(prop.size().size() == 2);
    REQUIRE(prop.size()[0] == 11);
    REQUIRE(prop.size()[1] == 11);
    REQUIRE(prop(0, 0) == double{});
    prop(0, 0) = 3;
    REQUIRE(prop(0, 0) == 3);
  }
  SECTION("chunked") {
    auto& prop = g.add_chunked_vertex_property<double>("double_prop");
    REQUIRE(prop.size().size() == 2);
    REQUIRE(prop.size()[0] == 11);
    REQUIRE(prop.size()[1] == 11);
    REQUIRE(prop(0, 0) == double{});
    prop(0, 0) = 3;
    REQUIRE(prop(0, 0) == 3);
  }
}
//==============================================================================
TEST_CASE("grid_vertex_property_sampler", "[grid][sampler]") {
  grid  g{linspace{0.0, 1.0, 2}, linspace{0.0, 1.0, 2}};
  auto& prop    = g.add_contiguous_vertex_property<double>("double_prop");
  prop(0,0) = 1;
  prop(1,0) = 2;
  prop(0,1) = 3;
  prop(1,1) = 4;
  auto  sampler = prop.sampler<interpolation::linear, interpolation::linear>();
  REQUIRE(sampler(0, 0) == 1);
  REQUIRE(sampler(1, 0) == 2);
  REQUIRE(sampler(0, 1) == 3);
  REQUIRE(sampler(1, 1) == 4);

  REQUIRE(sampler(0.5, 0) == 1.5);
  REQUIRE(sampler(0.25, 0) == 1.25);
  REQUIRE(sampler(0.5, 1) == 3.5);
  REQUIRE(sampler(0.25, 1) == 3.25);
  REQUIRE(sampler(0, 0.5) == 2);
  REQUIRE(sampler(0, 0.25) == 1.5);
  REQUIRE(sampler(1, 0.5) == 3);
  REQUIRE(sampler(1, 0.25) == 2.5);
  REQUIRE(sampler(0.5, 0.5) == 2.5);
}
//==============================================================================
TEST_CASE("grid_vertex_prop_cubic", "[grid][sampler][cubic]") {
  std::array                           dim0{0.0, 1.0, 2.0};
  std::array                           dim1{0.0, 1.0};
  grid<decltype(dim0), decltype(dim1)> g{dim0, dim1};

  auto& u =
      g.add_chunked_vertex_property<double, x_fastest>("u", std::vector<size_t>{2, 2});
  auto u_sampler = u.sampler<interpolation::cubic, interpolation::linear>();
  u(0, 0) = 0;
  u(1, 0) = 6;
  u(2, 0) = 2;

  u(0, 1) = 4;
  u(1, 1) = 0;
  u(2, 1) = 4;

  double const y = 0.25;
  REQUIRE(u_sampler(0, y) == Approx(1));
  REQUIRE(u_sampler(1, y) == Approx(4.5));
  REQUIRE(u_sampler(2, y) == Approx(2.5));
}
//==============================================================================
TEST_CASE("grid_chunked_vertex_property", "[grid][vertex][chunked][property]") {
  std::array                                           dim0{0, 1, 2};
  std::vector                                          dim1{0, 1, 2};
  linspace                                             dim2{0.0, 2.0, 3};
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};

  auto& u_prop =
      g.add_chunked_vertex_property<double, x_fastest>(
          "u", std::vector<size_t>{2, 2, 2});

  REQUIRE(u_prop(0, 0, 0) == 0);
  u_prop(0, 0, 0) = 1;
  REQUIRE(u_prop(0, 0, 0) == 1);

  REQUIRE(u_prop(0, 1, 2) == 0);
  u_prop(0, 1, 2) = 3;
  REQUIRE(u_prop(0, 1, 2) == 3);

  REQUIRE_NOTHROW(g.vertex_property<double>("u"));
  REQUIRE_THROWS(g.vertex_property<float>("u"));
  REQUIRE_THROWS(g.vertex_property<float>("v"));

  auto& v_prop = g.add_contiguous_vertex_property<float, x_fastest>("v");
  REQUIRE(v_prop(0, 0, 0) == 0);
  v_prop(0, 0, 0) = 1;
  REQUIRE(v_prop(0, 0, 0) == 1);

  REQUIRE(v_prop(0, 1, 2) == 0);
  v_prop(0, 1, 2) = 3;
  REQUIRE(v_prop(0, 1, 2) == 3);

  REQUIRE_NOTHROW(g.vertex_property<double>("u"));
  REQUIRE_NOTHROW(g.vertex_property<float>("v"));
  REQUIRE_THROWS(g.vertex_property<double>("v"));
}
//==============================================================================
//TEST_CASE("grid_sample_1d_cubic", "[grid][sampler][1d][cubic]") {
//  linspace            dim{0.0, 1.0, 11};
//  grid<decltype(dim)> g{dim};
//  using prop_value_type = double;
//
//  std::string const prop_name = "u";
//  auto&             u_prop =
//      g.add_contiguous_vertex_property<prop_value_type, x_fastest,
//                                       interpolation::cubic>(prop_name);
//  u_prop.container().at(4) = 5;
//  u_prop.container().at(5) = 6;
//  REQUIRE(u_prop.sample(0.45) > 5);
//  REQUIRE(u_prop.sample(0.45) < 6);
//}
//==============================================================================
// TEST_CASE("grid_face_property", "[grid][face_property]") {
//  std::array  dim0{-1.0, 1.0, 2.0, 5.0, 6.0};
//  std::vector dim1{-3.0, 0.0, 2.0, 3.0, 4.0};
//  grid        g{dim0, dim1};
//
//  // add property on vertical grid edges (index = 0)
//  auto& u_prop = g.add_face_property<double, 0>("u");
//  // add chunked property on horizontal grid edges (index = 1, chunk size =
//  128) auto& v_prop = g.add_chunked_face_property<double, 1>("v");
//
//  // set some values
//  u_prop.container().at(0, 0) = 3;
//  u_prop.container().at(1, 0) = 5;
//  u_prop.container().at(0, 1) = 6;
//  u_prop.container().at(1, 1) = 8;
//
//  // sample in cell
//  SECTION("on data points") {
//    REQUIRE(u_prop.sample(-1, -1.5) == Approx(3));
//    REQUIRE(u_prop.sample(1, -1.5) == Approx(5));
//    REQUIRE(u_prop.sample(-1, 1) == Approx(6));
//    REQUIRE(u_prop.sample(1, 1) == Approx(8));
//  }
//
//  SECTION("on x-edge") {
//    REQUIRE(u_prop.sample(0, -1.5) == Approx(4));
//    REQUIRE(u_prop.sample(0, 1) == Approx(7));
//  }
//
//  SECTION("on y-edge") {
//    REQUIRE(u_prop.sample(-1, -0.25) == Approx(4.5));
//    REQUIRE(u_prop.sample(1, -0.25) == Approx(6.5));
//  }
//
//  SECTION("center") { REQUIRE(u_prop.sample(0, -0.25) == Approx(5.5)); }
//
//  v_prop.container().at(0, 0) = 4;
//  v_prop.container().at(1, 0) = 6;
//}
//==============================================================================
//TEST_CASE("grid_lazy_netcdf", "[grid][lazy][netcdf]") {
//  std::string const file_path     = "simple_xy.nc";
//  std::string const dim0_name     = "x";
//  std::string const dim1_name     = "y";
//  std::string const variable_name = "data";
//  // We are reading 2D data, a 8 x 6 grid.
//  size_t constexpr NX = 8;
//  size_t constexpr NY = 6;
//
//  std::vector<double> data_out(NX * NY);
//  // create some data
//  for (size_t j = 0; j < NY; ++j) {
//    for (size_t i = 0; i < NX; ++i) {
//      size_t idx    = i + NX * j;
//      data_out[idx] = idx;
//    }
//  }
//  data_out[4 + NX * 0] = 0;
//  data_out[5 + NX * 0] = 0;
//  data_out[4 + NX * 1] = 0;
//  data_out[5 + NX * 1] = 0;
//  linspace dim0{0.0, 1.0, NX};
//  linspace dim1{0.0, 1.0, NY};
//
//  netcdf::file f_out{file_path, netCDF::NcFile::replace};
//  auto         dim0_netcdf = f_out.add_dimension(dim0_name, NX);
//  auto         dim1_netcdf = f_out.add_dimension(dim1_name, NY);
//  f_out.add_variable<double>(dim0_name, dim0_netcdf).write(dim0);
//  f_out.add_variable<double>(dim1_name, dim1_netcdf).write(dim1);
//  f_out.add_variable<double>(variable_name, {dim1_netcdf, dim0_netcdf})
//      .write(data_out);
//
//  grid<decltype(dim0), decltype(dim1)> g{dim0, dim1};
//  auto&                                prop =
//      g.add_vertex_property<netcdf::lazy_reader<double>, interpolation::linear,
//                            interpolation::linear>(
//          "prop", file_path, variable_name, std::vector<size_t>{2, 2});
//
//  [[maybe_unused]] auto& prop_via_file =
//      g.add_vertex_property<netcdf::lazy_reader<double>, interpolation::linear,
//                            interpolation::linear>(
//          "prop_via_file",
//          netcdf::file{file_path, netCDF::NcFile::read}.variable<double>(
//              variable_name),
//          std::vector<size_t>{2, 2});
//
//  prop.sample(0.5, 0.5);
//}
//==============================================================================
TEST_CASE("grid_amira_write", "[grid][amira]") {
  linspace                                             dim0{0.0, 1.0, 3};
  linspace                                             dim1{0.0, 1.0, 3};
  linspace                                             dim2{0.0, 1.0, 3};
  grid<decltype(dim0), decltype(dim1), decltype(dim2)> g{dim0, dim1, dim2};
  auto& prop       = g.add_contiguous_vertex_property<vec<double, 2>>("bla");
  prop.at(1, 1, 1) = vec{1, 1};
  g.write_amira("amira_prop.am", prop);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
