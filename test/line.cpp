//#include <tatooine/analytical/fields/numerical/doublegyre.h>
//#include <tatooine/analytical/fields/numerical/sinuscosinus.h>
//#include <tatooine/constant_vectorfield.h>
//#include <tatooine/interpolation.h>
#include <tatooine/line.h>
//#include <tatooine/ode/vclibs/rungekutta43.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("line_push_back", "[line][push_back]") {
  line2 l;
  l.push_back(vec{0.0, 0.0});
  l.push_back(vec{1.0, 1.0});
  l.push_back(vec{2.0, 0.0});
  REQUIRE(approx_equal(l.vertex_at(0), vec{0, 0}, 0));
  REQUIRE(approx_equal(l.vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(l.vertex_at(2), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE("line_initializer_list", "[line][initializer_list]") {
  line2 l{vec{0, 0}, vec{1, 1}, vec{2, 0}};
  REQUIRE(approx_equal(l.vertex_at(0), vec{0, 0}, 0));
  REQUIRE(approx_equal(l.vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(l.vertex_at(2), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE("line_push_front", "[line][push_front]") {
  line2 l;
  l.push_front(vec{0, 0});
  l.push_front(vec{1, 1});
  l.push_front(vec{2, 0});
  REQUIRE(approx_equal(l.vertex_at(2), vec{0, 0}, 0));
  REQUIRE(approx_equal(l.vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(l.vertex_at(0), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE("line_tangent_container", "[line][property]") {
  line2 l;
  l.push_front(vec{0, 0});
  l.push_front(vec{1, 1});
  l.push_front(vec{2, 0});
  auto &prop = l.scalar_vertex_property("prop");
}
////==============================================================================
//TEST_CASE("line_tangent_container", "[line][tangents][container]") {
//  line<double, 2> l;
//  l.push_front(vec{0, 0});
//  l.push_front(vec{1, 1});
//  l.push_front(vec{2, 0});
//  std::vector<vec<double, 2>> tangents(3);
//  boost::copy(l.tangents(), begin(tangents));
//}
////==============================================================================
//TEST_CASE("line_parameterized_initialization",
//          "[line][parameterization][initialization]") {
//  parameterized_line<double, 2, interpolation::linear> l{
//      {vec{1, 2}, 0}, {vec{2, 3}, 1}, {vec{3, 4}, 2}};
//  auto const& x0 = l.vertex_at(0);
//  auto const& x1 = l.vertex_at(1);
//  auto const& x2 = l.vertex_at(2);
//  auto const& t0 = l.parameterization_at(0);
//  auto const& t1 = l.parameterization_at(1);
//  auto const& t2 = l.parameterization_at(2);
//  REQUIRE(x0(0) == 1);
//  REQUIRE(x0(1) == 2);
//  REQUIRE(t0 == 0);
//
//  REQUIRE(x1(0) == 2);
//  REQUIRE(x1(1) == 3);
//  REQUIRE(t1 == 1);
//
//  REQUIRE(x2(0) == 3);
//  REQUIRE(x2(1) == 4);
//  REQUIRE(t2 == 2);
//}
////==============================================================================
//TEST_CASE("line_sampling_linear",
//          "[line][parameterization][linear][sampling]") {
//  vec                                                  v0{0.1, 0.2};
//  vec                                                  v1{0.5, 0.9};
//  vec                                                  v2{0.9, 0.2};
//  parameterized_line<double, 2, interpolation::linear> l;
//  l.push_back(v0, 0);
//  l.push_back(v1, 0.5);
//  l.push_back(v2, 1);
//
//  REQUIRE(approx_equal(l.sample(0), v0));
//  REQUIRE(approx_equal(l.sample(0.5), v1));
//  REQUIRE(approx_equal(l.sample(1), v2));
//
//  REQUIRE(approx_equal(l.sample(0.1 / 2), (v0 * 0.9 + v1 * 0.1)));
//  REQUIRE(approx_equal(l.sample(0.9 / 2), (v0 * 0.1 + v1 * 0.9)));
//  REQUIRE(approx_equal(l.sample(1.1 / 2), (v1 * 0.9 + v2 * 0.1)));
//  REQUIRE(approx_equal(l.sample(1.9 / 2), (v1 * 0.1 + v2 * 0.9)));
//  REQUIRE_NOTHROW(l.sample(0.3));
//  REQUIRE_NOTHROW(l.sample(0.7));
//  REQUIRE_THROWS(l.sample(-0.01));
//  REQUIRE_THROWS(l.sample(1.01));
//}
////==============================================================================
//TEST_CASE("line_sampling_cubic",
//          "[line][parameterization][cubic][sampling]") {
//  vec                                                  v0{0.0, 0.0};
//  vec                                                  v1{1.0, 1.0};
//  vec                                                  v2{2.0, 0.0};
//  parameterized_line<double, 2, interpolation::linear> l;
//  l.push_back(v0, 0);
//  l.push_back(v1, 1);
//  l.push_back(v2, 2);
//
//  REQUIRE(approx_equal(l.sample(0), v0));
//  REQUIRE(approx_equal(l.sample(1), v1));
//  REQUIRE(approx_equal(l.sample(2), v2));
//  REQUIRE_NOTHROW(l.sample(0.5));
//  REQUIRE_NOTHROW(l.sample(1.5));
//  REQUIRE_THROWS(l.sample(-0.01));
//  REQUIRE_THROWS(l.sample(2.01));
//}
////==============================================================================
//TEST_CASE("line_paramaterization_quadratic_tangent",
//          "[line][parameterization][quadratic][tangent]") {
//  SECTION("simple") {
//    vec                                                  v0{1.0, 1.0};
//    double                                               t0 = 0;
//    vec                                                  v1{3.0, 2.0};
//    double                                               t1 = 3;
//    vec                                                  v2{2.0, 3.0};
//    double                                               t2 = 4;
//    parameterized_line<double, 2, interpolation::linear> l{
//        {v0, t0}, {v1, t1}, {v2, t2}};
//
//    l.tangents_to_property();
//    // l.write_vtk("simple_quadratic_tangents.vtk");
//  }
//}
////==============================================================================
//TEST_CASE("line_paramaterization_uniform",
//          "[line][parameterization][uniform]") {
//  vec                                                  v0{0.0, 0.0};
//  vec                                                  v1{1.0, 1.0};
//  vec                                                  v2{2.0, 0.0};
//  parameterized_line<double, 2, interpolation::linear> l;
//  l.push_back(v0, 0);
//  l.push_back(v1, 0);
//  l.push_back(v2, 0);
//  l.uniform_parameterization();
//}
////==============================================================================
//TEST_CASE("line_paramaterization_chordal",
//          "[line][parameterization][chordal]") {
//  vec                                                  v0{0.0, 0.0};
//  vec                                                  v1{1.0, 1.0};
//  vec                                                  v2{2.0, 0.0};
//  parameterized_line<double, 2, interpolation::linear> l;
//  l.push_back(v0, 0);
//  l.push_back(v1, 0);
//  l.push_back(v2, 0);
//  l.chordal_parameterization();
//}
////==============================================================================
//TEST_CASE("line_paramaterization_centripetal",
//          "[line][parameterization][centripetal]") {
//  vec                                                  v0{0.0, 0.0};
//  vec                                                  v1{1.0, 1.0};
//  vec                                                  v2{2.0, 0.0};
//  parameterized_line<double, 2, interpolation::linear> l;
//  l.push_back(v0, 0);
//  l.push_back(v1, 0);
//  l.push_back(v2, 0);
//  l.centripetal_parameterization();
//}
////==============================================================================
//TEST_CASE("line_resample", "[line][parameterization][resample]") {
//  using integral_curve_t =
//      parameterized_line<double, 2, interpolation::cubic>;
//  using vertex_idx = integral_curve_t::vertex_idx;
//  SECTION("double gyre pathline") {
//    analytical::fields::numerical::doublegyre v;
//    ode::vclibs::rungekutta43<double, 2>      rk43;
//    integral_curve_t                          integral_curve;
//    rk43.solve(v, vec{0.2, 0.2}, 0, 10, [&](auto const& y, auto const t) {
//      integral_curve.push_back(y, t);
//    });
//    auto& curvature = integral_curve.add_vertex_property<double>("curvature");
//    for (size_t i = 0; i < integral_curve.num_vertices(); ++i) {
//      curvature[vertex_idx{i}] =
//          integral_curve.curvature(integral_curve.parameterization_at(i));
//    }
//    integral_curve.write_vtk("original_dg_pathline.vtk");
//    size_t i = 0;
//    for (auto n : std::array{10000, 20000}) {
//      integral_curve.resample(linspace(0.0, 10.0, n))
//          .write_vtk("resampled_dg_pathline" + std::to_string(i++) + ".vtk");
//    }
//  }
//  SECTION("simple line") {
//    SECTION("linear") {
//      parameterized_line<double, 2, interpolation::linear> l{
//          {vec{0.0, 0.0}, 0}, {vec{1.0, 1.0}, 1}, {vec{2.0, 0.0}, 2}};
//      l.write_vtk("original_line.vtk");
//      l.resample(linspace(0.0, 2.0, 10000))
//          .write_vtk("resampled_line_linear.vtk");
//    }
//    SECTION("cubic") {
//      parameterized_line<double, 2, interpolation::cubic> l{
//          {vec{0.0, 0.0}, 0}, {vec{1.0, 1.0}, 1}, {vec{2.0, 0.0}, 2}};
//      l.write_vtk("original_line.vtk");
//      auto  l2        = l.resample(linspace(0.0, 2.0, 10000));
//      auto& curvature = l2.add_vertex_property<double>("curvature");
//      for (size_t i = 0; i < l2.num_vertices(); ++i) {
//        curvature[vertex_idx{i}] = l2.curvature(l2.parameterization_at(i));
//      }
//      l2.write_vtk("resampled_line_cubic.vtk");
//    }
//  }
//}
////==============================================================================
//TEST_CASE("line_curvature", "[line][parameterization][curvature]") {
//  parameterized_line<double, 2, interpolation::cubic> l;
//  l.push_back(vec{0.0, 0.0}, 0);
//  l.push_back(vec{1.0, 1.0}, 1);
//  l.push_back(vec{2.0, 0.0}, 2);
//  REQUIRE(!std::isnan(l.curvature(0)));
//  REQUIRE(!std::isnan(l.curvature(1)));
//  REQUIRE(!std::isnan(l.curvature(2)));
//  REQUIRE(!std::isinf(l.curvature(0)));
//  REQUIRE(!std::isinf(l.curvature(1)));
//  REQUIRE(!std::isinf(l.curvature(2)));
//}
////==============================================================================
//TEST_CASE("line_curvature2", "[line][parameterization][curvature1]") {
//  using integral_curve_t =
//      parameterized_line<double, 2, interpolation::cubic>;
//  analytical::fields::numerical::doublegyre v;
//  ode::vclibs::rungekutta43<double, 2>      ode;
//  integral_curve_t                          integral_curve;
//  auto& tangents = integral_curve.tangents_property();
//  ode.solve(v, vec{0.1, 0.1}, 5, 6,
//            [&]( auto const& y, auto const t,auto const& dy) {
//              integral_curve.push_back(y, t, false);
//              tangents.back() = dy;
//            });
//  ode.solve(v, vec{0.1, 0.1}, 5, -6,
//            [&](auto const& y, auto const t,auto const& dy) {
//              integral_curve.push_front(y, t, false);
//              tangents.front() = dy;
//            });
//  integral_curve.update_interpolators();
//
//  for (size_t i = 0; i < integral_curve.num_vertices(); ++i) {
//    CAPTURE(i);
//    CAPTURE(integral_curve.num_vertices());
//    CAPTURE(integral_curve.parameterization_at(i));
//    CAPTURE(integral_curve.vertex_at(i));
//    CAPTURE(integral_curve.tangent_at(i));
//    CAPTURE(
//        v(integral_curve.vertex_at(i), integral_curve.parameterization_at(i)));
//    REQUIRE(approx_equal(
//        integral_curve.tangent_at(i),
//        v(integral_curve.vertex_at(i), integral_curve.parameterization_at(i))));
//  }
//  double tfront = integral_curve.front_parameterization();
//  double tback  = integral_curve.back_parameterization();
//  REQUIRE(tfront == Approx(-1).margin(1e-10));
//  REQUIRE(tback == Approx(11).margin(1e-10));
//  {
//    INFO("curve:\n" << integral_curve.interpolator(tfront).curve());
//    INFO("tfront: " << tfront);
//    INFO("pos:    " << integral_curve(tfront));
//    INFO("vec:    " << v(integral_curve(tfront), tfront));
//    INFO("tang:   " << integral_curve.tangent(tfront));
//    const auto curv = integral_curve.curvature(tfront);
//    REQUIRE(!std::isnan(curv));
//    REQUIRE(!std::isinf(curv));
//  }
//}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
