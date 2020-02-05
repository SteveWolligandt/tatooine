#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/streamsurface.h>
#include <tatooine/doublegyre.h>
#include <tatooine/sinuscosinus.h>
//#include <tatooine/boussinesq.h>
#include <tatooine/spacetime_field.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
using interpolation::hermite;
using interpolation::linear;

TEST_CASE(
    "streamsurface_simple_spacetime_doublegyre",
    "[streamsurface][simple][numerical][doublegyre][dg][spacetime_field]") {
  numerical::doublegyre v;
  spacetime_field       vst{v};
  streamsurface         ssf{
      vst,
      0,
      parameterized_line<double, 3, hermite>{
          {{0.1, 0.2, 0.0}, 0.0},
          {{0.5, 0.9, 0.0}, 0.5},
          {{0.9, 0.2, 0.0}, 1.0}},
      integration::vclibs::rungekutta43<double, 3, hermite>{},
  };
  ssf.discretize<simple_discretization>(5ul, 1.0, 0.0, 1.0)
      .write_vtk("streamsurface_dg_simple.vtk");
}
//==============================================================================
TEST_CASE(
    "streamsurface_hultquist_spacetime_doublegyre",
    "[streamsurface][hultquist][numerical][doublegyre][dg][spacetime_field]") {
  numerical::doublegyre v;
  spacetime_field       vst{v};
  streamsurface         ssf{
      vst, 0,
      parameterized_line<double, 3, hermite>{
          {{0.1, 0.2, 0.0}, 0.0},
          {{0.5, 0.9, 0.0}, 0.5},
          {{0.9, 0.2, 0.0}, 1.0}},
      integration::vclibs::rungekutta43<double, 3, hermite>{}};
  ssf.discretize<hultquist_discretization>(20ul, 0.1, -20.0, 20.0)
      .write_vtk("streamsurface_dg_hultquist.vtk");
}
//==============================================================================
TEST_CASE(
    "streamsurface_simple_sinuscosinus",
    "[streamsurface][simple][numerical][sinuscosinus][sc]") {
  numerical::sinuscosinus v;
  parameterized_line<double, 2, linear> seed{{{0.0, 0.0}, 0.0},
                                                            {{1.0, 0.0}, 1.0}};
  integration::vclibs::rungekutta43<double, 2, linear>
                integrator;
  streamsurface ssf{v, 0.0, seed, integrator};
  ssf.discretize<simple_discretization>(2ul, 0.01, -M_PI, M_PI)
      .write_vtk("streamsurface_sc_simple.vtk");
}
//==============================================================================
TEST_CASE(
    "streamsurface_hultquist_sinuscosinus",
    "[streamsurface][hultquist][numerical][sinuscosinus][sc]") {
  numerical::sinuscosinus                              v;
  parameterized_line<double, 2, linear> seed{{{0.0, 0.0}, 0.0},
                                                            {{1.0, 0.0}, 1.0}};

  integration::vclibs::rungekutta43<double, 2, linear>
                integrator;
  streamsurface ssf{v, 0.0, seed, integrator};
  ssf.discretize<hultquist_discretization>(2ul, 0.01, -M_PI, M_PI)
      .write_vtk("streamsurface_sc_hultquist.vtk");
}

////==============================================================================
//TEST_CASE("[streamsurface] out of domain integration", "[streamsurface]") {
//  UnsteadyGridSamplerVF<2, double, 2, hermite, hermite, linear> testvector2c(
//      "testvector2c.am");
//  FixedTimeVectorfield fixed{testvector2c, 0};
//
//  auto bb     = testvector2c.sampler().boundingbox();
//  auto center = bb.center();
//  std::cout << bb << '\n' << center << '\n';
//  line<double, 2> seed{
//      {vec{bb.min(0) * 0.5 + center(0) * 0.5, center(1)}, 0.0},
//      {vec{bb.min(0) * 0.5 + center(0) * 0.5, bb.min(1) + 1e-3}, 1.0},
//  };
//  streamsurface ssf{fixed, 0, seed};
//  auto          discretization = ssf.discretize(4, 0.1, 0, 1);
//  discretization.write_vtk("testvector2c_ssf.vtk");
//}
//
////==============================================================================
//TEST_CASE("[streamsurface] unsteady out of domain integration",
//          "[streamsurface]") {
//  UnsteadyGridSamplerVF<2, double, 2, hermite, hermite, linear> testvector2c(
//      "testvector2c.am");
//
//  grid          g{linspace{-0.9, -0.1, 26}, linspace{0.1, 0.9, 26}};
//  streamsurface ssf{testvector2c, 0, grid_edge{g.at(2, 24), g.at(1, 23)}};
//  ssf.discretize(4, 0.05, -0.5, 0).write_vtk("unsteady_testvector2c_ssf.vtk");
//}
//
////==============================================================================
//TEST_CASE("[streamsurface] out of domain integration2", "[streamsurface]") {
//  struct vf_t : vectorfield<2, double, vf_t> {
//    using parent_t = vectorfield<2, double, vf_t>;
//    using parent_t::out_of_domain;
//    using parent_t::pos_t;
//    using parent_t::vec_t;
//    constexpr bool in_domain(const pos_t& x, double) const {
//      return x(1) >= -0.65;
//    }
//    constexpr vec_t evaluate(const pos_t& x, double t) const {
//      if (!in_domain(x, t)) throw out_of_domain{x, t};
//      return {0, -1};
//    }
//  } vf;
//
//  SECTION("center") {
//    line<double, 2> seed{{vec{-0.5, 0}, 0.0},
//                         {vec{-0.25, -0.25}, 1.0},
//                         {vec{0, -0.5}, 2.0},
//                         {vec{0.25, -0.25}, 3.0},
//                         {vec{0.5, 0}, 4.0}};
//    streamsurface   ssf{vf, 0, seed};
//    auto            discretization = ssf.discretize(5, 0.1, 0, 0.8);
//    discretization.write_vtk("bounded_down_center_ssf.vtk");
//  }
//  SECTION("left") {
//    line<double, 2> seed{
//        {vec{0, -0.5}, 2.0}, {vec{0.25, -0.25}, 3.0}, {vec{0.5, 0}, 4.0}};
//    streamsurface ssf{vf, 0, seed};
//    auto          discretization = ssf.discretize(5, 0.1, 0, 0.7);
//    discretization.write_vtk("bounded_down_left_ssf.vtk");
//  }
//  SECTION("right") {
//    line<double, 2> seed{
//        {vec{-0.5, 0}, 0.0}, {vec{-0.25, -0.25}, 1.0}, {vec{0, -0.5}, 2.0}};
//    streamsurface ssf{vf, 0, seed};
//    auto          discretization = ssf.discretize(5, 0.1, 0, 0.7);
//    discretization.write_vtk("bounded_down_right_ssf.vtk");
//  }
//}
//
////==============================================================================
//TEST_CASE("[streamsurface] out of domain integration3", "[streamsurface]") {
//  struct vf_t : vectorfield<2, double, vf_t> {
//    using parent_t = vectorfield<2, double, vf_t>;
//    using parent_t::out_of_domain;
//    using parent_t::pos_t;
//    using parent_t::vec_t;
//    constexpr bool in_domain(const pos_t& x, double) const {
//      return x(0) >= 0 && x(1) >= 0;
//    }
//    constexpr vec_t evaluate(const pos_t& x, double t) const {
//      if (!in_domain(x, t)) throw out_of_domain{x, t};
//      return {-1, -1};
//    }
//  } vf;
//
//  line<double, 2> seed{{vec{0.1, 1}, 0.0}, {vec{1, 0.1}, 1.0}};
//  streamsurface   ssf{vf, 0, seed};
//  auto            discretization = ssf.discretize(5, 0.1, 0, 2);
//  discretization.write_vtk("border_corner.vtk");
//}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
