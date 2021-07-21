#include <tatooine/rectilinear_grid.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("grid_ctor_raw_size", "[rectilinear_grid][ctor][raw_size]") {
  size_t const res_x = 10;
  size_t const res_y = 11;
  size_t const res_z = 12;
  auto         g     = rectilinear_grid{res_x, res_y, res_z};
  REQUIRE(g.num_dimensions() == 3);
  REQUIRE(is_linspace<std::decay_t<decltype(g.dimension(0))>>);
  REQUIRE(is_linspace<std::decay_t<decltype(g.dimension(1))>>);
  REQUIRE(is_linspace<std::decay_t<decltype(g.dimension(2))>>);
  REQUIRE(g.dimension(0).front() == 0);
  REQUIRE(g.dimension(1).front() == 0);
  REQUIRE(g.dimension(2).front() == 0);
  REQUIRE(g.dimension(0).back() == 1);
  REQUIRE(g.dimension(1).back() == 1);
  REQUIRE(g.dimension(2).back() == 1);
  REQUIRE(g.dimension(0).size() == 10);
  REQUIRE(g.dimension(1).size() == 11);
  REQUIRE(g.dimension(2).size() == 12);
}
//==============================================================================
TEST_CASE("grid_copy_constructor", "[rectilinear_grid][copy][constructor]") {
  auto const dim0       = std::array{0, 1, 2};
  auto const dim1       = std ::array{0, 1, 2};
  auto       g0         = rectilinear_grid{dim0, dim1};
  auto&      prop       = g0.insert_scalar_vertex_property("prop");
  auto const some_value = 100;
  prop(0, 0)            = some_value;
  auto  g1              = g0;
  auto& prop_copy       = g1.vertex_property<double>("prop");
  REQUIRE(prop(0, 0) == prop_copy(0, 0));
  prop(0, 0) = 0;
  REQUIRE_FALSE(prop(0, 0) == prop_copy(0, 0));
}
//==============================================================================
TEST_CASE("grid_vertex_indexing", "[rectilinear_grid][vertex][indexing]") {
  auto const dim0 = std::array{0, 1, 2};
  auto const dim1 = std::vector{0, 1, 2};
  auto const dim2 = linspace{0.0, 1.0, 11};
  auto const g    = rectilinear_grid{dim0, dim1, dim2};
  auto const v000 = g.vertex_at(0, 0, 0);
  auto const v111 = g.vertex_at(1, 1, 1);
  auto const v221 = g.vertex_at(2, 2, 1);

  REQUIRE(approx_equal(v000, vec{0.0, 0.0, 0.0}));
  REQUIRE(approx_equal(v111, vec{1.0, 1.0, 0.1}));
  REQUIRE(approx_equal(v221, vec{2.0, 2.0, 0.1}));
}
//==============================================================================
TEST_CASE("grid_vertex_iterator", "[rectilinear_grid][vertex][iterator]") {
  auto g  = rectilinear_grid{std::array{0.0, 1.0, 2.0}, std::vector{0.0, 1.0, 2.0},
                linspace{0.0, 2.0, 3}};
  auto it = begin(vertices(g));
  REQUIRE(approx_equal(g[*it], g.vertex_at(0, 0, 0)));
  REQUIRE(approx_equal(g[*next(it)], g.vertex_at(1, 0, 0)));
  REQUIRE(approx_equal(g[*next(it, 3)], g.vertex_at(0, 1, 0)));
  REQUIRE(approx_equal(g[*next(it, 9)], g.vertex_at(0, 0, 1)));
  // REQUIRE(next(it, 27) == end(gv));
}
//==============================================================================
TEST_CASE("grid_cell_index", "[rectilinear_grid][cell_index]") {
  auto const dim0            = std::array{0.0, 1.2, 1.3};
  auto const dim1            = std::vector{0, 1, 2, 4};
  auto const dim2            = linspace{0.0, 2.0, 3};
  auto const dim3            = linspace{0.0, 1.0, 25};
  auto       g               = rectilinear_grid{dim0, dim1, dim2, dim3};
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
  // REQUIRE(factor6 == 1);
}
//==============================================================================
TEST_CASE("grid_vertex_property", "[rectilinear_grid][property]") {
  auto g = rectilinear_grid{linspace{0.0, 1.0, 11}, linspace{0.0, 1.0, 11}};
  SECTION("contiguous") {
    auto& prop = g.insert_contiguous_vertex_property<double>("double_prop");
    REQUIRE(prop.size().size() == 2);
    REQUIRE(prop.size()[0] == 11);
    REQUIRE(prop.size()[1] == 11);
    REQUIRE(prop(0, 0) == double{});
    prop(0, 0) = 3;
    REQUIRE(prop(0, 0) == 3);
  }
  SECTION("chunked") {
    auto& prop = g.insert_chunked_vertex_property<double>("double_prop");
    REQUIRE(prop.size().size() == 2);
    REQUIRE(prop.size()[0] == 11);
    REQUIRE(prop.size()[1] == 11);
    REQUIRE(prop(0, 0) == double{});
    prop(0, 0) = 3;
    REQUIRE(prop(0, 0) == 3);
  }
}
//==============================================================================
TEST_CASE("grid_vertex_property_sampler", "[rectilinear_grid][sampler][linear]") {
  auto  g    = rectilinear_grid{linspace{0.0, 10.0, 11}, linspace{0.0, 10.0, 11}};
  auto& prop = g.insert_scalar_vertex_property("double_prop");
  prop(0, 0) = 1;
  prop(1, 0) = 2;
  prop(0, 1) = 3;
  prop(1, 1) = 4;
  REQUIRE(prop(0, 0) == 1);
  REQUIRE(prop(1, 0) == 2);
  REQUIRE(prop(0, 1) == 3);
  REQUIRE(prop(1, 1) == 4);
  auto sampler = prop.linear_sampler();
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
TEST_CASE("grid_vertex_prop_cubic", "[rectilinear_grid][sampler][cubic]") {
  auto const dim0 = std::array{0.0, 1.0, 2.0};
  auto const dim1 = std::array{0.0, 1.0};
  auto       g    = rectilinear_grid{dim0, dim1};

  auto& u         = g.insert_scalar_vertex_property("u");
  auto  u_sampler = u.cubic_sampler();

  u(0, 1) = 4;
  u(1, 1) = 0;
  u(2, 1) = 4;
  u(0, 0) = 0;
  u(1, 0) = 6;
  u(2, 0) = 2;

  auto  resample_grid = rectilinear_grid{linspace{0.0, 2.0, 201}, linspace{0.0, 1.0, 101}};
  auto& resampled_u   = resample_grid.insert_scalar_vertex_property("u");
  resample_grid.vertices().iterate_indices([&](auto const... is) {
    resampled_u(is...) = u_sampler(resample_grid.vertex_at(is...));
  });

  g.write_vtk("source_u.vtk");
  resample_grid.write_vtk("resampled_u.vtk");

  double const y = 0.25;
  REQUIRE(u_sampler(0, y) == Approx(1));
  REQUIRE(u_sampler(1, y) == Approx(4.5));
  REQUIRE(u_sampler(2, y) == Approx(2.5));
}
//==============================================================================
TEST_CASE("grid_chunked_vertex_property", "[rectilinear_grid][vertex][chunked][property]") {
  auto const dim0 = std::array{0, 1, 2};
  auto const dim1 = std::vector{0, 1, 2};
  auto const dim2 = linspace{0.0, 2.0, 3};
  auto       g    = rectilinear_grid{dim0, dim1, dim2};

  auto& u_prop = g.insert_chunked_vertex_property<double, x_fastest>(
      "u", std::vector<size_t>{2, 2, 2});

  REQUIRE(u_prop(0, 0, 0) == 0);
  u_prop(0, 0, 0) = 1;
  REQUIRE(u_prop(0, 0, 0) == 1);

  REQUIRE(u_prop(0, 1, 2) == 0);
  u_prop(0, 1, 2) = 3;
  REQUIRE(u_prop(0, 1, 2) == 3);

  REQUIRE_NOTHROW(g.vertex_property<double>("u"));
  REQUIRE_THROWS(g.vertex_property<float>("u"));

  auto& v_prop = g.insert_contiguous_vertex_property<float, x_fastest>("v");
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
TEST_CASE("grid_amira_write", "[rectilinear_grid][amira]") {
  auto const dim0  = linspace{0.0, 1.0, 3};
  auto const dim1  = linspace{0.0, 1.0, 3};
  auto const dim2  = linspace{0.0, 1.0, 3};
  auto       g     = rectilinear_grid{dim0, dim1, dim2};
  auto&      prop  = g.insert_contiguous_vertex_property<vec<double, 2>>("bla");
  prop.at(1, 1, 1) = vec{1, 1};
  g.write_amira("amira_prop.am", prop);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
