#include <tatooine/doublegyre.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("integration_vclibs_rk43",
          "[integration][vclibs][rk43][rungekutta43][doublegyre][symbolic]") {
  symbolic::doublegyre dg;
  integration::vclibs::rungekutta43<double, 2> rk43;
  rk43.integrate(dg, {0.1, 0.1}, 0, 30);
  rk43.integrate(dg, {0.1, 0.1}, 0, -30).write_vtk("dg_pathline.vtk");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
