#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/ftle.h>
#include <tatooine/grid_sampler.h>
#include <tatooine/doublegyre.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("ftle", "[ftle][doublegyre][dg]") {
  numerical::doublegyre dg;
  ftle dg_ftle{dg, integration::vclibs::rungekutta43<double, 2>{}, 10};
  auto dg_ftle_grid = resample<interpolation::linear, interpolation::linear>(
      dg_ftle, grid{linspace{0.0, 2.0, 200}, linspace{0.0, 1.0, 100}}, 0);
  dg_ftle_grid.sampler().write_png("dg_ftle.png");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
