#include <tatooine/doublegyre.h>
#include <tatooine/sinuscosinus.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("integration_vclibs_dg_rk43",
          "[integration][vclibs][rk43][rungekutta43][doublegyre][symbolic]") {
  symbolic::doublegyre                         dg;
  integration::vclibs::rungekutta43<double, 2> rk43;
  rk43.integrate(dg, {0.1, 0.1}, 0, 30);
  rk43.integrate(dg, {0.1, 0.1}, 0, -30).write_vtk("dg_pathline.vtk");
}

//==============================================================================
TEST_CASE("integration_vclibs_stdg_rk43",
          "[integration][vclibs][rk43][rungekutta43][doublegyre][symbolic]"
          "[spacetime_field]") {
  symbolic::doublegyre                         dg;
  spacetime_field                              stdg{dg};
  integration::vclibs::rungekutta43<double, 3> rk43;
  rk43.integrate(stdg, {0.1, 0.1, 0.0}, 0, 30);
  rk43.integrate(stdg, {0.1, 0.1, 0.0}, 0, -30)
      .write_vtk("spacetime_dg_pathline_0_1.vtk");
  rk43.integrate(stdg, {0.2, 0.2, 0.0}, 0, 30);
  rk43.integrate(stdg, {0.2, 0.2, 0.0}, 0, -30)
      .write_vtk("spacetime_dg_pathline_0_2.vtk");
}

//==============================================================================
TEST_CASE("integration_vclibs_sincos_rk43",
          "[integration][vclibs][rk43][rungekutta43][sincos][sc]"
          "[symbolic]") {
  numerical::sinuscosinus                      v;
  integration::vclibs::rungekutta43<double, 2> rk43;
  auto pl = rk43.integrate(v, {0, 0}, 0, -M_PI, M_PI);
  pl.write_vtk("sincos_pathline.vtk");

  parameterized_line<double, 2> resampled;
  for (auto t: linspace(-M_PI, M_PI, 20)) {
    resampled.push_back(pl(t), t);
  }
  resampled.write_vtk("sincos_resampled_pathline.vtk");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
