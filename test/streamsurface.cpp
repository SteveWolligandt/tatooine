#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/analytical/numerical/sinuscosinus.h>
#include <tatooine/streamsurface.h>
#include <tatooine/spacetime_vectorfield.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using interpolation::cubic;
using interpolation::linear;
//==============================================================================
TEST_CASE("streamsurface_spacetime_doublegyre_sampling",
          "[streamsurface][numerical][doublegyre][dg][sample]") {
  using seedcurve_type = line<double, 2>;
  auto const v         = analytical::numerical::doublegyre{};
  auto       seedcurve = seedcurve_type{};
  seedcurve.push_back(0.1, 0.1);
  seedcurve.parameterization().back() = 0;
  seedcurve.push_back(0.1, 0.9);
  seedcurve.parameterization().back() = 1;
  auto ssf=streamsurface {flowmap(v), -1, 1, seedcurve};

  auto sampler = seedcurve.cubic_sampler();
  {
    CAPTURE(ssf(0, -1));
    CAPTURE(sampler(0));
    REQUIRE(approx_equal(ssf(0, -1), sampler(0)));
  }
  {
    CAPTURE(ssf(0.5, 0));
    CAPTURE(sampler(0.5));
    REQUIRE(approx_equal(ssf(0.5, 0), sampler(0.5)));
  }
  {
    CAPTURE(ssf(1, 1));
    CAPTURE(sampler(1));
    REQUIRE(approx_equal(ssf(1, 1), sampler(1)));
  }
  {
    CAPTURE(ssf(0, 0));
    CAPTURE(sampler(0));
    REQUIRE(approx_equal(ssf(0, 0), sampler(0)));
  }
}
//==============================================================================
TEST_CASE(
    "streamsurface_hultquist_spacetime_doublegyre",
    "[streamsurface][hultquist][numerical][doublegyre][dg][spacetime_field]") {
  auto v         = analytical::numerical::doublegyre{};
  auto vst       = spacetime_vectorfield(v);
  auto seedcurve = line3{};
  seedcurve.push_back(0.1, 0.2, 0.0);
  seedcurve.parameterization().back() = 0;
  seedcurve.push_back(0.5, 0.9, 0.0);
  seedcurve.parameterization().back() = 0.5;
  seedcurve.push_back(0.9, 0.2, 0.0);
  seedcurve.parameterization().back() = 1;
  auto ssf                            = streamsurface{flowmap(vst), seedcurve};
  ssf.discretize<hultquist_discretization>(10UL, 0.1, -2.0, 2.0);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
