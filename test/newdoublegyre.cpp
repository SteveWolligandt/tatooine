#include <tatooine/newdoublegyre.h>
#include <tatooine/linspace.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

//TEST_CASE("newdoublegyre_bifurcationline", "[newdoublegyre][ndg][bifurcationline]") {
//  numerical::newdoublegyre ndg;
//  symbolic::newdoublegyre  sdg;
//
//  for (auto t : linspace{0.0, 10.0, 100}) {
//    REQUIRE(sdg.bifurcationline(t) == Approx(ndg.bifurcationline(t)));
//  }
//}

//==============================================================================
TEST_CASE("newdoublegyre_timeoffset", "[newdoublegyre][ndg][timeoffset]") {
  numerical::newdoublegyre ndg;
  symbolic::newdoublegyre  sdg;

  for (auto t : linspace{0.0, 10.0, 100}) {
    REQUIRE(sdg.timeoffset(t) == Approx(ndg.timeoffset(t)));
  }
}

//==============================================================================
TEST_CASE("newdoublegyre_sampling", "[newdoublegyre][ndg][symbolic][numerical]") {
  numerical::newdoublegyre ndg;
  symbolic::newdoublegyre  sdg;
  for (auto x : linspace{0.0, 2.0, 21}) {
    for (auto y : linspace{0.0, 1.0, 11}) {
      for (auto t : linspace{0.0, 10.0, 11}) {
        auto vs = sdg({x, y}, t);
        auto vn = ndg({x, y}, t);
        for (size_t i = 0; i < 2; ++i) {
          REQUIRE(std::abs(vs(i) - vn(i)) < 1e-5);
        }
      }
    }
  }
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
