#include <catch2/catch.hpp>

#include <tatooine/doublegyre.h>
#include <tatooine/grid_sampler.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("grid_sampler_singletime",
          "[grid_sampler][doublegyre][dg][symbolic][singletime]") {
  using namespace tatooine::numerical;
  // using namespace tatooine::symbolic;
  using namespace tatooine::interpolation;

  const double t = 0;
  doublegyre   dg;
  auto         dgr = resample<hermite, hermite>(
      dg, grid{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}}, t);

  REQUIRE(dgr.sampler().boundingbox().min(0) == 0.0);
  REQUIRE(dgr.sampler().boundingbox().max(0) == 2.0);
  REQUIRE(dgr.sampler().boundingbox().min(1) == 0.0);
  REQUIRE(dgr.sampler().boundingbox().max(1) == 1.0);
  REQUIRE(dgr.sampler().data().size(0) == 21);
  REQUIRE(dgr.sampler().data().size(1) == 11);

  for (auto v : dgr.sampler().vertices()) {
    auto x = v.pos();
    INFO("position: " << x);
    auto vel_dg = dg(x, t);
    INFO("doublegyre: " << vel_dg);
    auto vel_dgr = dgr(x, t);
    INFO("resampled: " << vel_dgr);
    REQUIRE(approx_equal(vel_dg, vel_dgr, 1e-15));
  }
}

TEST_CASE("grid_sampler_multitime",
          "[grid_sampler][doublegyre][dg][symbolic][multitime]") {
  using namespace tatooine::numerical;
  // using namespace tatooine::symbolic;
  using namespace tatooine::interpolation;

  doublegyre   dg;
  auto         dgr = resample<hermite, hermite, linear>(
      dg, grid{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}},
      linspace{0.0, 10.0, 101});

  REQUIRE(dgr.sampler().boundingbox().min(0) == 0.0);
  REQUIRE(dgr.sampler().boundingbox().min(1) == 0.0);
  REQUIRE(dgr.sampler().boundingbox().min(2) == 0.0);
  REQUIRE(dgr.sampler().boundingbox().max(0) == 2.0);
  REQUIRE(dgr.sampler().boundingbox().max(1) == 1.0);
  REQUIRE(dgr.sampler().boundingbox().max(2) == 10.0);
  REQUIRE(dgr.sampler().data().size(0) == 21);
  REQUIRE(dgr.sampler().data().size(1) == 11);
  REQUIRE(dgr.sampler().data().size(2) == 101);

  for (auto v : dgr.sampler().vertices()) {
    auto   gridpos = v.pos();
    vec    x{gridpos(0), gridpos(1)};
    auto t = gridpos(2);
    INFO("position: " << x);
    auto vel_dg = dg(x, t);
    INFO("doublegyre: " << vel_dg);
    auto vel_dgr = dgr(x, t);
    INFO("resampled: " << vel_dgr);
    REQUIRE(approx_equal(vel_dg, vel_dgr, 1e-15));
  }
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
