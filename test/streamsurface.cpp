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
//==============================================================================
TEST_CASE(
    "streamsurface_spacetime_doublegyre_sampling",
    "[streamsurface][numerical][doublegyre][dg][sample]") {
  using integrator_t  = integration::vclibs::rungekutta43<double, 2, hermite>;
  using seedcurve_t   = parameterized_line<double, 2, linear>;
  numerical::doublegyre v;
  const seedcurve_t     seedcurve{{{0.1, 0.1}, 0.0}, {{0.1, 0.9}, 1.0}};
  streamsurface         ssf{v, -1, 1, seedcurve, integrator_t{}};

  {
    const double u = 0;
    ssf.streamline_at(u, 0, 10);
    REQUIRE(ssf.streamline_at(u).front_parameterization() == -1);
    REQUIRE(ssf.streamline_at(u).back_parameterization() == Approx(9));
    REQUIRE(approx_equal(ssf.streamline_at(u).front_vertex(), seedcurve(u)));
    REQUIRE_FALSE(approx_equal(ssf.streamline_at(u).back_vertex(), seedcurve(u)));
  }{
    const double u = 1;
    ssf.streamline_at(u, -10, 0);
    REQUIRE(ssf.streamline_at(u).front_parameterization() == -9);
    REQUIRE(ssf.streamline_at(u).back_parameterization() == Approx(1));
    REQUIRE_FALSE(approx_equal(ssf.streamline_at(u).front_vertex(), seedcurve(u)));
    REQUIRE(approx_equal(ssf.streamline_at(u).back_vertex(), seedcurve(u)));
  }{
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
    REQUIRE_FALSE(approx_equal(ssf(0, 0), seedcurve(0)));
  }
}
//==============================================================================
TEST_CASE(
    "streamsurface_doublegyre_caching",
    "[streamsurface][numerical][doublegyre][dg][caching]") {
  using integrator_t  = integration::vclibs::rungekutta43<double, 2, hermite>;
  using seedcurve_t   = parameterized_line<double, 2, linear>;
  numerical::doublegyre v;
  const seedcurve_t     seedcurve{{{0.1, 0.1}, 0.0}, {{0.1, 0.9}, 1.0}};
  streamsurface         ssf{v, -1, 1, seedcurve, integrator_t{}};

  for (auto u : linspace(0.0, 1.0, 3)) {
    CAPTURE(u, ssf.t0(u));
    ssf(u, ssf.t0(u), -1, 1);
    REQUIRE(ssf.streamline_at(u, 0, 0).front_parameterization() ==
            ssf.t0(u) - 1);
    REQUIRE(ssf.streamline_at(u, 0, 0).back_parameterization() ==
            ssf.t0(u) + 1);
  }
}
//==============================================================================
TEST_CASE(
    "streamsurface_doublegyre_front_evolving",
    "[streamsurface][numerical][doublegyre][dg][evaluate_vectorfield]") {
  using integrator_t  = integration::vclibs::rungekutta43<double, 2, hermite>;
  using seedcurve_t   = parameterized_line<double, 2, linear>;
  numerical::doublegyre v;
  const seedcurve_t     seedcurve{{{0.1, 0.5}, 0.0}, {{0.9, 0.5}, 1.0}};
  SECTION("time zero") {
    streamsurface ssf{v, seedcurve, integrator_t{}};
    CHECK(approx_equal(ssf.vectorfield_at(0, 0), v(seedcurve(0), 0), 0));
    CHECK(approx_equal(ssf.vectorfield_at(0.5, 0), v(seedcurve(0.5), 0), 0));
    CHECK(approx_equal(ssf.vectorfield_at(1, 0), v(seedcurve(1), 0), 0));
  }
  SECTION("single time") {
    streamsurface ssf{v, 1, seedcurve, integrator_t{}};
    CHECK(approx_equal(ssf.vectorfield_at(0, 1), v(seedcurve(0), 1), 0));
    CHECK(approx_equal(ssf.vectorfield_at(0.5, 1), v(seedcurve(0.5), 1), 0));
    CHECK(approx_equal(ssf.vectorfield_at(1, 1), v(seedcurve(1), 1), 0));

    CHECK_FALSE(approx_equal(ssf.vectorfield_at(0, 0), v(seedcurve(0), 0), 0));
    CHECK_FALSE(approx_equal(ssf.vectorfield_at(0.5, 0), v(seedcurve(0.5), 0), 0));
    CHECK_FALSE(approx_equal(ssf.vectorfield_at(1, 0), v(seedcurve(1), 0), 0));

    CHECK_FALSE(approx_equal(ssf.vectorfield_at(0, 2), v(seedcurve(0), 2), 0));
    CHECK_FALSE(approx_equal(ssf.vectorfield_at(0.5, 2), v(seedcurve(0.5), 2), 0));
    CHECK_FALSE(approx_equal(ssf.vectorfield_at(1, 2), v(seedcurve(1), 2), 0));
  }
  SECTION("varying time") {
    streamsurface ssf{v, -1, 1, seedcurve, integrator_t{}};
    CHECK(approx_equal(ssf.vectorfield_at(0, -1), v(seedcurve(0), -1), 0));
    CHECK(approx_equal(ssf.vectorfield_at(0.5, 0), v(seedcurve(0.5), 0), 0));
    CHECK(approx_equal(ssf.vectorfield_at(1, 1), v(seedcurve(1), 1), 0));
    CHECK(approx_equal(ssf.vectorfield_at(0, 0), v(ssf(0, 0), 0), 0));

    CHECK_FALSE(approx_equal(ssf.vectorfield_at(0, 0), v(seedcurve(0), 0), 0));
    CHECK_FALSE(approx_equal(ssf.vectorfield_at(1, 0), v(seedcurve(1), 0), 0));
  }
}
//==============================================================================
TEST_CASE(
    "streamsurface_simple_spacetime_doublegyre",
    "[streamsurface][simple][numerical][doublegyre][dg][spacetime_field]") {
  numerical::doublegyre v;
  spacetime_field       vst{v};
  streamsurface         ssf{
      vst, 0, 2,
      parameterized_line<double, 3, linear>{{{0.1, 0.1, 0.0}, 0.0},
                                            {{0.1, 0.9, 0.0}, 1.0}},
      integration::vclibs::rungekutta43<double, 3, hermite>{},
  };
  ssf.discretize<simple_discretization>(5UL, 0.1, -10.0, 10.0)
      .write_vtk("streamsurface_dg_simple.vtk");
}
//==============================================================================
TEST_CASE(
    "streamsurface_hultquist_spacetime_doublegyre",
    "[streamsurface][hultquist][numerical][doublegyre][dg][spacetime_field]") {
  numerical::doublegyre v;
  spacetime_field       vst{v};
  streamsurface         ssf{
      vst, -2, 2,
      parameterized_line<double, 3, hermite>{{{0.1, 0.2, 0.0}, 0.0},
                                             {{0.5, 0.9, 0.0}, 0.5},
                                             {{0.9, 0.2, 0.0}, 1.0}},
      integration::vclibs::rungekutta43<double, 3, hermite>{}};
  ssf.discretize<hultquist_discretization>(20UL, 0.1, -20.0, 20.0)
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
  ssf.discretize<simple_discretization>(2UL, 0.01, -M_PI, M_PI)
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
  ssf.discretize<hultquist_discretization>(2UL, 0.01, -M_PI, M_PI)
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
