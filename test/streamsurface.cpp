#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/analytical/numerical/sinuscosinus.h>
#include <tatooine/streamsurface.h>
//#include <tatooine/boussinesq.h>
#include <tatooine/spacetime_vectorfield.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using interpolation::cubic;
using interpolation::linear;
//==============================================================================
TEST_CASE("streamsurface_spacetime_doublegyre_sampling",
          "[streamsurface][numerical][doublegyre][dg][sample]") {
  using seedcurve_t  = parameterized_line<double, 2, linear>;
  analytical::numerical::doublegyre v;
  const seedcurve_t seedcurve{{{0.1, 0.1}, 0.0}, {{0.1, 0.9}, 1.0}};
  streamsurface     ssf{v, -1, 1, seedcurve};
  using streamsurface_t = decltype(ssf);
  REQUIRE(std::is_same_v<streamsurface_t::flowmap_t, decltype(flowmap(v))>);

  {
    CAPTURE(ssf(0, -1));
    CAPTURE(seedcurve(0));
    REQUIRE(approx_equal(ssf(0, -1), seedcurve(0)));
  }
  {
    CAPTURE(ssf(0.5, 0));
    CAPTURE(seedcurve(0.5));
    REQUIRE(approx_equal(ssf(0.5, 0), seedcurve(0.5)));
  }
  {
    CAPTURE(ssf(1, 1));
    CAPTURE(seedcurve(1));
    REQUIRE(approx_equal(ssf(1, 1), seedcurve(1)));
  }
  {
    CAPTURE(ssf(0, 0));
    CAPTURE(seedcurve(0));
    REQUIRE(approx_equal(ssf(0, 0), seedcurve(0)));
  }
}
//==============================================================================
//TEST_CASE(
//    "streamsurface_simple_spacetime_doublegyre",
//    "[streamsurface][simple][numerical][doublegyre][dg][spacetime_field]") {
//  analytical::numerical::doublegyre v;
//  spacetime_field                           vst{v};
//  streamsurface                             ssf{
//      vst,
//      parameterized_line<double, 3, cubic>{{{0.1, 0.2, 0.0}, 0.0},
//                                             {{0.5, 0.9, 0.0}, 0.5},
//                                             {{0.9, 0.2, 0.0}, 1.0}},
//      integration::vclibs::rungekutta43<double, 3, cubic>{}};
//  ssf.discretize<simple_discretization>(20UL, 0.1, -20.0, 20.0)
//      .write_vtk("streamsurface_dg_simple.vtk");
//}
//==============================================================================
TEST_CASE(
    "streamsurface_hultquist_spacetime_doublegyre",
    "[streamsurface][hultquist][numerical][doublegyre][dg][spacetime_field]") {
  analytical::numerical::doublegyre v;
  spacetime_vectorfield                           vst{v};
  streamsurface                             ssf{
      vst, parameterized_line<double, 3, cubic>{{{0.1, 0.2, 0.0}, 0.0},
                                                  {{0.5, 0.9, 0.0}, 0.5},
                                                  {{0.9, 0.2, 0.0}, 1.0}}};
  ssf.discretize<hultquist_discretization>(100UL, 0.1, -20.0, 20.0)
      .write_vtk("streamsurface_dg_hultquist.vtk");
}
//==============================================================================
TEST_CASE(
    "streamsurface_schulze_spacetime_doublegyre",
    "[streamsurface][schulze][numerical][doublegyre][dg][spacetime_field]") {
  analytical::numerical::doublegyre v;
  streamsurface                             ssf{
      v, parameterized_line<double, 2, cubic>{{{0.45, 0.2}, 0.0},
                                                {{0.55, 0.2}, 1.0}}};
  ssf.discretize<schulze_discretization>(10UL, 20)
      .write_vtk("streamsurface_dg_schulze.vtk");
}
//==============================================================================
//TEST_CASE("streamsurface_simple_sinuscosinus",
//          "[streamsurface][simple][numerical][sinuscosinus][sc]") {
//  analytical::numerical::sinuscosinus          v;
//  parameterized_line<double, 2, linear>                seed{{{0.0, 0.0}, 0.0},
//                                             {{1.0, 0.0}, 1.0}};
//  integration::vclibs::rungekutta43<double, 2, linear> integrator;
//  streamsurface ssf{v, 0.0, seed, integrator};
//  ssf.discretize<simple_discretization>(2UL, 0.01, -M_PI, M_PI)
//      .write_vtk("streamsurface_sc_simple.vtk");
//}
////==============================================================================
//TEST_CASE("streamsurface_hultquist_sinuscosinus",
//          "[streamsurface][hultquist][numerical][sinuscosinus][sc]") {
//  analytical::numerical::sinuscosinus v;
//  parameterized_line<double, 2, linear>       seed{{{0.0, 0.0}, 0.0},
//                                             {{1.0, 0.0}, 1.0}};
//
//  integration::vclibs::rungekutta43<double, 2, linear> integrator;
//  streamsurface ssf{v, 0.0, seed, integrator};
//  ssf.discretize<hultquist_discretization>(2UL, 0.01, -M_PI, M_PI)
//      .write_vtk("streamsurface_sc_hultquist.vtk");
//}
//
//////==============================================================================
//// TEST_CASE("[streamsurface] out of domain integration", "[streamsurface]") {
////  UnsteadyGridSamplerVF<2, double, 2, cubic, cubic, linear> testvector2c(
////      "testvector2c.am");
////  FixedTimeVectorfield fixed{testvector2c, 0};
////
////  auto bb     = testvector2c.sampler().boundingbox();
////  auto center = bb.center();
////  std::cout << bb << '\n' << center << '\n';
////  line<double, 2> seed{
////      {vec{bb.min(0) * 0.5 + center(0) * 0.5, center(1)}, 0.0},
////      {vec{bb.min(0) * 0.5 + center(0) * 0.5, bb.min(1) + 1e-3}, 1.0},
////  };
////  streamsurface ssf{fixed, 0, seed};
////  auto          discretization = ssf.discretize(4, 0.1, 0, 1);
////  discretization.write_vtk("testvector2c_ssf.vtk");
////}
////
//////==============================================================================
//// TEST_CASE("[streamsurface] unsteady out of domain integration",
////          "[streamsurface]") {
////  UnsteadyGridSamplerVF<2, double, 2, cubic, cubic, linear> testvector2c(
////      "testvector2c.am");
////
////  grid          g{linspace{-0.9, -0.1, 26}, linspace{0.1, 0.9, 26}};
////  streamsurface ssf{testvector2c, 0, grid_edge{g.at(2, 24), g.at(1, 23)}};
////  ssf.discretize(4, 0.05, -0.5, 0).write_vtk("unsteady_testvector2c_ssf.vtk");
////}
////
//////==============================================================================
//// TEST_CASE("[streamsurface] out of domain integration2", "[streamsurface]") {
////  struct vf_t : vectorfield<2, double, vf_t> {
////    using parent_t = vectorfield<2, double, vf_t>;
////    using parent_t::out_of_domain;
////    using parent_t::pos_type;
////    using parent_t::vec_t;
////    constexpr bool in_domain(const pos_type& x, double) const {
////      return x(1) >= -0.65;
////    }
////    constexpr vec_t evaluate(const pos_type& x, double t) const {
////      if (!in_domain(x, t)) throw out_of_domain{x, t};
////      return {0, -1};
////    }
////  } vf;
////
////  SECTION("center") {
////    line<double, 2> seed{{vec{-0.5, 0}, 0.0},
////                         {vec{-0.25, -0.25}, 1.0},
////                         {vec{0, -0.5}, 2.0},
////                         {vec{0.25, -0.25}, 3.0},
////                         {vec{0.5, 0}, 4.0}};
////    streamsurface   ssf{vf, 0, seed};
////    auto            discretization = ssf.discretize(5, 0.1, 0, 0.8);
////    discretization.write_vtk("bounded_down_center_ssf.vtk");
////  }
////  SECTION("left") {
////    line<double, 2> seed{
////        {vec{0, -0.5}, 2.0}, {vec{0.25, -0.25}, 3.0}, {vec{0.5, 0}, 4.0}};
////    streamsurface ssf{vf, 0, seed};
////    auto          discretization = ssf.discretize(5, 0.1, 0, 0.7);
////    discretization.write_vtk("bounded_down_left_ssf.vtk");
////  }
////  SECTION("right") {
////    line<double, 2> seed{
////        {vec{-0.5, 0}, 0.0}, {vec{-0.25, -0.25}, 1.0}, {vec{0, -0.5}, 2.0}};
////    streamsurface ssf{vf, 0, seed};
////    auto          discretization = ssf.discretize(5, 0.1, 0, 0.7);
////    discretization.write_vtk("bounded_down_right_ssf.vtk");
////  }
////}
////
//////==============================================================================
//// TEST_CASE("[streamsurface] out of domain integration3", "[streamsurface]") {
////  struct vf_t : vectorfield<2, double, vf_t> {
////    using parent_t = vectorfield<2, double, vf_t>;
////    using parent_t::out_of_domain;
////    using parent_t::pos_type;
////    using parent_t::vec_t;
////    constexpr bool in_domain(const pos_type& x, double) const {
////      return x(0) >= 0 && x(1) >= 0;
////    }
////    constexpr vec_t evaluate(const pos_type& x, double t) const {
////      if (!in_domain(x, t)) throw out_of_domain{x, t};
////      return {-1, -1};
////    }
////  } vf;
////
////  line<double, 2> seed{{vec{0.1, 1}, 0.0}, {vec{1, 0.1}, 1.0}};
////  streamsurface   ssf{vf, 0, seed};
////  auto            discretization = ssf.discretize(5, 0.1, 0, 2);
////  discretization.write_vtk("border_corner.vtk");
////}
//
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
