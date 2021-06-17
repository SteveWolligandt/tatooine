//#include <tatooine/analytical/fields/numerical/doublegyre.h>
//#include <tatooine/analytical/fields/numerical/sinuscosinus.h>
//#include <tatooine/constant_vectorfield.h>
#include <tatooine/line.h>
//#include <tatooine/ode/vclibs/rungekutta43.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE_METHOD(line2, "line_push_back", "[line][push_back]") {
  push_back(vec{0.0, 0.0});
  push_back(vec{1.0, 1.0});
  push_back(vec{2.0, 0.0});
  REQUIRE(approx_equal(vertex_at(0), vec{0, 0}, 0));
  REQUIRE(approx_equal(vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(vertex_at(2), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_push_front", "[line][push_front]") {
  push_front(vec{0, 0});
  push_front(vec{1, 1});
  push_front(vec{2, 0});
  REQUIRE(approx_equal(vertex_at(2), vec{0, 0}, 0));
  REQUIRE(approx_equal(vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(vertex_at(0), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE("line_initializer_list", "[line][initializer_list]") {
  line2 l{vec{0, 0}, vec{1, 1}, vec{2, 0}};
  REQUIRE(approx_equal(l.vertex_at(0), vec{0, 0}, 0));
  REQUIRE(approx_equal(l.vertex_at(1), vec{1, 1}, 0));
  REQUIRE(approx_equal(l.vertex_at(2), vec{2, 0}, 0));
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_property", "[line][property]") {
  push_front(vec{0, 0});
  push_front(vec{1, 1});
  push_front(vec{2, 0});
  auto &prop             = scalar_vertex_property("prop");
  prop[vertex_handle{0}] = 1;
  prop[vertex_handle{1}] = 2;
  prop[vertex_handle{2}] = 3;
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_tangents", "[line][tangents]") {
  push_back(vec2{0, 0});
  push_back(vec2{1, 1});
  push_back(vec2{2, 0});
  compute_parameterization();
  normalize_parameterization();
  compute_tangents();
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_linear_sampler", "[line][linear][sampler]") {
  push_back(vec2{0, 0});
  push_back(vec2{1, 1});
  push_back(vec2{2, 0});
  compute_uniform_parameterization();
  auto const x = linear_sampler();
  REQUIRE(approx_equal(x(0.0), vec2{0, 0}));
  REQUIRE(approx_equal(x(0.5), vec2{0.5, 0.5}));
  REQUIRE(approx_equal(x(0.75), vec2{0.75, 0.75}));
  REQUIRE(approx_equal(x(1.0), vec2{1, 1}));
  REQUIRE(approx_equal(x(2.0), vec2{2, 0}));
}
//==============================================================================
TEST_CASE_METHOD(line2, "line_resample", "[line][parameterization][resample]") {
  push_back(vec2{0, 0});
  push_back(vec2{1, 1});
  push_back(vec2{2, 0});
  auto &prop      = scalar_vertex_property("prop");
  using handle    = line2::vertex_handle;
  prop[handle{0}] = 1;
  prop[handle{1}] = 2;
  prop[handle{2}] = 30;
  compute_uniform_parameterization();

  write_vtk("original_line.vtk");
  resample<interpolation::linear>(linspace{0.0, 2.0, 101})
      .write_vtk("line_linear_resampled.vtk");
  resample<interpolation::cubic>(linspace{0.0, 2.0, 101})
      .write_vtk("line_cubic_resampled.vtk");
}
////==============================================================================
// TEST_CASE("line_curvature", "[line][parameterization][curvature]") {
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
// TEST_CASE("line_curvature2", "[line][parameterization][curvature1]") {
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
//        v(integral_curve.vertex_at(i),
//        integral_curve.parameterization_at(i)));
//    REQUIRE(approx_equal(
//        integral_curve.tangent_at(i),
//        v(integral_curve.vertex_at(i),
//        integral_curve.parameterization_at(i))));
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
