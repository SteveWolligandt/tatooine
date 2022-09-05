#include <tatooine/rectilinear_grid.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("rectilinear_grid_constructors",
          "[rectilinear_grid][ctor][constructors]") {
  SECTION("default") {
    auto g = uniform_rectilinear_grid2{};
    REQUIRE(g.num_dimensions() == 2);
    REQUIRE(g.size<0>() == 0);
    REQUIRE(g.size<1>() == 0);
  }
  SECTION("copy") {
    auto const dim0       = std::array{0.0, 1.0, 2.0};
    auto const dim1       = std::array{0.0, 1.0, 2.0};
    auto       g0         = rectilinear_grid{dim0, dim1};
    auto&      prop       = g0.scalar_vertex_property("prop");
    auto const some_value = 100;
    prop(0, 0)            = some_value;
    auto  g1              = g0;
    auto& prop_copy       = g1.vertex_property<double>("prop");
    REQUIRE(prop(0, 0) == prop_copy(0, 0));
    prop(0, 0) = 0;
    REQUIRE_FALSE(prop(0, 0) == prop_copy(0, 0));
  }
  SECTION("move") {
    auto const  dim0       = std::vector{0.0, 1.0, 2.0};
    auto const  dim1       = std::vector{0.0, 1.0, 2.0};
    auto        g0         = rectilinear_grid{dim0, dim1};
    auto const* dim0_data  = g0.dimension<0>().data();
    auto const* dim1_data  = g0.dimension<1>().data();
    auto&       prop       = g0.scalar_vertex_property("prop");
    auto const  some_value = 100;
    prop(0, 0)             = some_value;
    auto const* prop_data  = &prop(0, 0);
    REQUIRE(prop_data[0] == 100);

    auto g1 = std::move(g0);
    REQUIRE(dim0_data == g1.dimension<0>().data());
    REQUIRE(dim1_data == g1.dimension<1>().data());
    REQUIRE(prop_data == &g1.scalar_vertex_property("prop")(0, 0));
  }
  SECTION("aabb") {
    auto g = rectilinear_grid{aabb2{vec2{0, -1}, vec2{1, 2}}, 12, 13};
    REQUIRE(g.num_dimensions() == 2);
    REQUIRE(g.size<0>() == 12);
    REQUIRE(g.size<1>() == 13);
  }
  SECTION("raw size") {
    auto const res_x = std::size_t(10);
    auto const res_y = std::size_t(11);
    auto const res_z = std::size_t(12);
    auto       g     = rectilinear_grid{res_x, res_y, res_z};
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
}
//==============================================================================
TEST_CASE("rectilinear_grid_vertex_indexing",
          "[rectilinear_grid][vertex][indexing]") {
  auto const dim0 = std::array{0.0, 1.0, 2.0};
  auto const dim1 = std::vector{0.0, 1.0, 2.0};
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
TEST_CASE("rectilinear_grid_vertex_iterator",
          "[rectilinear_grid][vertex][iterator]") {
  auto g  = rectilinear_grid{std::array{0.0, 1.0, 2.0},
                            std::vector{0.0, 1.0, 2.0}, linspace{0.0, 2.0, 3}};
  auto it = begin(vertices(g));
  REQUIRE(approx_equal(g[*it], g.vertex_at(0, 0, 0)));
  REQUIRE(approx_equal(g[*next(it)], g.vertex_at(1, 0, 0)));
  REQUIRE(approx_equal(g[*next(it, 3)], g.vertex_at(0, 1, 0)));
  REQUIRE(approx_equal(g[*next(it, 9)], g.vertex_at(0, 0, 1)));
  // REQUIRE(next(it, 27) == end(gv));
}
//==============================================================================
TEST_CASE("rectilinear_grid_cell_index", "[rectilinear_grid][cell_index]") {
  auto const dim0            = std::array{0.0, 1.2, 1.3};
  auto const dim1            = std::vector{0.0, 1.0, 2.0, 4.0};
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
TEST_CASE("rectilinear_grid_vertex_property", "[rectilinear_grid][property]") {
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
TEST_CASE("rectilinear_grid_dimensions", "[rectilinear_grid][dimensions]") {
  SECTION("mixed get") {
    auto r = rectilinear_grid{std::vector<double>{1, 2, 3},
                              std::array<double, 3>{1, 2, 3},
                              linspace<double>{1, 3, 3}};
    REQUIRE(
        is_same<std::decay_t<decltype(r.dimension<0>())>, std::vector<double>>);
    REQUIRE(is_same<std::decay_t<decltype(r.dimension<1>())>,
                    std::array<double, 3>>);
    REQUIRE(
        is_same<std::decay_t<decltype(r.dimension<2>())>, linspace<double>>);
  }
  SECTION("dynamic") {
    auto const r = rectilinear_grid{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    for (std::size_t i = 0; i < 11; ++i) {
      REQUIRE(r.size(i) == 2);
    }
    REQUIRE(&r.dimension(0) == &r.dimension<0>());
    REQUIRE(&r.dimension(1) == &r.dimension<1>());
    REQUIRE(&r.dimension(2) == &r.dimension<2>());
    REQUIRE(&r.dimension(3) == &r.dimension<3>());
    REQUIRE(&r.dimension(4) == &r.dimension<4>());
    REQUIRE(&r.dimension(5) == &r.dimension<5>());
    REQUIRE(&r.dimension(6) == &r.dimension<6>());
    REQUIRE(&r.dimension(7) == &r.dimension<7>());
    REQUIRE(&r.dimension(8) == &r.dimension<8>());
    REQUIRE(&r.dimension(9) == &r.dimension<9>());
    REQUIRE(&r.dimension(10) == &r.dimension<10>());
  }
}
//==============================================================================
TEST_CASE("rectilinear_grid_topology",
          "[rectilinear_grid][topology]") {
  auto const r =
      rectilinear_grid{std::vector{1.0, 1.1, 2.0}, std::vector{2.0, 3.1, 4.0}};
  SECTION("min") {
    auto const min = r.min();
    REQUIRE(min(0) == 1.0);
    REQUIRE(min(1) == 2.0);
  }
  SECTION("max") {
    auto const max = r.max();
    REQUIRE(max(0) == 2.0);
    REQUIRE(max(1) == 4.0);
  }
  SECTION("center") {
    auto const center = r.center();
    REQUIRE(center(0) == Approx(1.5));
    REQUIRE(center(1) == Approx(3.0));
  }
  SECTION("extent") {
    auto const extent = r.extent();
    REQUIRE(extent(0) == Approx(1.0));
    REQUIRE(extent(1) == Approx(2.0));

    REQUIRE(r.extent<0>(0) == Approx(0.1));
    REQUIRE(r.extent<0>(1) == Approx(0.9));
    REQUIRE(r.extent<1>(0) == Approx(1.1));
    REQUIRE(r.extent<1>(1) == Approx(0.9));
  }
  SECTION("aabb") {
    auto const aabb = r.bounding_box();
    REQUIRE(aabb.min(0) == Approx(1.0));
    REQUIRE(aabb.min(1) == Approx(2.0));
    REQUIRE(aabb.max(0) == Approx(2.0));
    REQUIRE(aabb.max(1) == Approx(4.0));
  }
}
//==============================================================================
TEST_CASE("rectilinear_grid_push_back",
          "[rectilinear_grid][push_back]") {
  auto r =
      rectilinear_grid{std::vector{1.0, 1.1, 2.0}, std::vector{2.0, 3.5, 4.0}, linspace{0.0, 1.0, 11}};
  r.push_back<0>();
  REQUIRE(r.dimension<0>().back() == Approx(2.9));
  r.push_back<1>();
  REQUIRE(r.dimension<1>().back() == Approx(4.5));
  r.push_back<2>();
  REQUIRE(r.dimension<2>().back() == Approx(1.1));
}
//==============================================================================
TEST_CASE("rectilinear_grid_vertex_properties",
          "[rectilinear_grid][vertex_properties]") {
  SECTION("copy without property") {
    auto const prop_name = "prop";
    auto r = rectilinear_grid{32, 32};
    r.scalar_vertex_property(prop_name);
    auto r_copy = r.copy_without_properties();
    REQUIRE_FALSE(r_copy.has_vertex_property(prop_name));
  }

}
//==============================================================================
TEST_CASE("rectilinear_grid_vertex_property_sampler_scalar",
          "[rectilinear_grid][sampler][linear][scalar]") {
  auto  g = rectilinear_grid{linspace{0.0, 10.0, 11}, linspace{0.0, 10.0, 11}};
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
  REQUIRE(sampler(0.25, 0.25) == 1.75);
}
//==============================================================================
TEST_CASE("rectilinear_grid_vertex_property_sampler_vec",
          "[rectilinear_grid][sampler][linear][vec]") {
  auto  g = rectilinear_grid{linspace{0.0, 10.0, 11}, linspace{0.0, 10.0, 11}};
  auto& prop   = g.insert_vec2_vertex_property("double_prop");
  prop(0, 0)   = vec{1, 2};
  prop(1, 0)   = vec{2, 4};
  prop(0, 1)   = vec{3, 6};
  prop(1, 1)   = vec{4, 8};
  auto sampler = prop.linear_sampler();
  REQUIRE(sampler(0, 0)(0) == 1);
  REQUIRE(sampler(0, 0)(1) == 2);
  REQUIRE(sampler(1, 0)(0) == 2);
  REQUIRE(sampler(1, 0)(1) == 4);
  REQUIRE(sampler(0, 1)(0) == 3);
  REQUIRE(sampler(0, 1)(1) == 6);
  REQUIRE(sampler(1, 1)(0) == 4);
  REQUIRE(sampler(1, 1)(1) == 8);

  REQUIRE(sampler(0.5, 0)(0) == 1.5);
  REQUIRE(sampler(0.5, 0)(1) == 3);
  REQUIRE(sampler(0.25, 0)(0) == 1.25);
  REQUIRE(sampler(0.25, 0)(1) == 2.5);
  REQUIRE(sampler(0.5, 1)(0) == 3.5);
  REQUIRE(sampler(0.5, 1)(1) == 7);
  REQUIRE(sampler(0.25, 1)(0) == 3.25);
  REQUIRE(sampler(0.25, 1)(1) == 6.5);
  REQUIRE(sampler(0, 0.5)(0) == 2);
  REQUIRE(sampler(0, 0.5)(1) == 4);
  REQUIRE(sampler(0, 0.25)(0) == 1.5);
  REQUIRE(sampler(0, 0.25)(1) == 3);
  REQUIRE(sampler(1, 0.5)(0) == 3);
  REQUIRE(sampler(1, 0.5)(1) == 6);
  REQUIRE(sampler(1, 0.25)(0) == 2.5);
  REQUIRE(sampler(1, 0.25)(1) == 5);
  REQUIRE(sampler(0.5, 0.5)(0) == 2.5);
  REQUIRE(sampler(0.5, 0.5)(1) == 5);
  REQUIRE(sampler(0.25, 0.25)(0) == 1.75);
  REQUIRE(sampler(0.25, 0.25)(1) == 3.5);
}
//==============================================================================
TEST_CASE("rectilinear_grid_vertex_prop_cubic",
          "[rectilinear_grid][sampler][cubic]") {
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

  auto resample_grid =
      rectilinear_grid{linspace{0.0, 2.0, 201}, linspace{0.0, 1.0, 101}};
  auto& resampled_u = resample_grid.insert_scalar_vertex_property("u");
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
TEST_CASE("rectilinear_grid_chunked_vertex_property",
          "[rectilinear_grid][vertex][chunked][property]") {
  auto const dim0 = std::array{0.0, 1.0, 2.0};
  auto const dim1 = std::vector{0.0, 1.0, 2.0};
  auto const dim2 = linspace{0.0, 2.0, 3};
  auto       g    = rectilinear_grid{dim0, dim1, dim2};

  auto& u_prop = g.insert_chunked_vertex_property<double, x_fastest>(
      "u", std::vector<std::size_t>{2, 2, 2});

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
TEST_CASE("rectilinear_grid_vertex_property_diff_scalar",
          "[rectilinear_grid][vertex_property][diff][scalar]") {
  auto  grid        = rectilinear_grid{5, 6};
  auto& scalar      = grid.scalar_vertex_property("scalar");
  scalar(0,0) = 1;
  scalar(1,0) = 2;
  scalar(2,0) = 3;
  scalar(3,0) = 4;
  scalar(4,0) = 5;
  scalar(0,1) = 2;
  scalar(1,1) = 3;
  scalar(2,1) = 4;
  scalar(3,1) = 5;
  scalar(4,1) = 6;
  scalar(0,2) = 3;
  scalar(1,2) = 4;
  scalar(2,2) = 5;
  scalar(3,2) = 6;
  scalar(4,2) = 7;
  scalar(0,3) = 4;
  scalar(1,3) = 5;
  scalar(2,3) = 6;
  scalar(3,3) = 7;
  scalar(4,3) = 8;
  scalar(0,4) = 5;
  scalar(1,4) = 6;
  scalar(2,4) = 7;
  scalar(3,4) = 8;
  scalar(4,4) = 9;
  scalar(0,5) = 6;
  scalar(1,5) = 7;
  scalar(2,5) = 8;
  scalar(3,5) = 9;
  scalar(4,5) = 10;
  auto diff1_scalar = diff(scalar);
  for (std::size_t iy = 0; iy < 5; ++iy) {
    for (std::size_t ix = 0; ix < 5; ++ix) {
      REQUIRE(diff1_scalar(ix, iy)(0) == Approx(4));
      REQUIRE(diff1_scalar(ix, iy)(1) == Approx(5));
    }
  }
  auto diff2_scalar = diff(diff1_scalar);
  for (std::size_t i = 0; i < 5; ++i) {
    REQUIRE(diff2_scalar(i, i)(0, 0) == Approx(0).margin(1e-6));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
