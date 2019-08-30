#include <catch2/catch.hpp>

#include <tatooine/grid_sampler.h>
#include <tatooine/doublegyre.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("grid_sampler_time_slice", "[grid_sampler][doublegyre][dg][symbolic]") {
  using namespace tatooine::numerical;
  //using namespace tatooine::symbolic;
  using namespace tatooine::interpolation;

  const double t = 0;
  doublegyre dg;
  auto dgr = resample<hermite, hermite>(dg, grid{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}}, t);

  for (auto v : dgr.sampler().vertices()) {
    auto x = v.to_vec();
    INFO("position: " << x);
    auto vel_dg = dg(x, t);
    INFO("doublegyre: " << vel_dg);
    auto vel_dgr = dgr(x, t);
    INFO("resampled: " << vel_dgr);
    for (size_t i = 0; i < dg.tensor_dimension(0); ++i) {
      REQUIRE(vel_dg(i) == vel_dgr(i));
    }
  }
}


//==============================================================================
}  // namespace tatooine::test
//==============================================================================
