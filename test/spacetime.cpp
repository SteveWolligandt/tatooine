#include "../spacetime.h"
#include "../doublegyre.h"
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("spacetime_numerical_doublegyre", "[spacetime]") {
  numerical::doublegyre dg;
  spacetime stdg{dg};
  auto      dg_vel   = dg({0.3, 0.3}, 1);
  auto      stdg_vel = stdg({0.3, 0.3, 1.0});
  REQUIRE(dg_vel(0) == stdg_vel(0));
  REQUIRE(dg_vel(1) == stdg_vel(1));
  REQUIRE(stdg_vel(2) == 1);
}

//==============================================================================
TEST_CASE("spacetime_symbolic_doublegyre", "[spacetime]") {
  symbolic::doublegyre dg;
  spacetime stdg{dg};
  auto      dg_vel   = dg({0.3, 0.3}, 1);
  auto      stdg_vel = stdg({0.3, 0.3, 1.0});
  REQUIRE(dg_vel(0) == stdg_vel(0));
  REQUIRE(dg_vel(1) == stdg_vel(1));
  REQUIRE(stdg_vel(2) == 1);
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
