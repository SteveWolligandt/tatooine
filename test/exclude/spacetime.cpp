#include <tatooine/spacetime_field.h>
#include <tatooine/doublegyre.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("spacetime_field_numerical_doublegyre", "[spacetime_field]") {
  numerical::doublegyre dg;
  spacetime_field stdg{dg};
  auto      dg_vel   = dg({0.3, 0.3}, 1);
  auto      stdg_vel = stdg({0.3, 0.3, 1.0});
  REQUIRE(dg_vel(0) == stdg_vel(0));
  REQUIRE(dg_vel(1) == stdg_vel(1));
  REQUIRE(stdg_vel(2) == 1);
}

//==============================================================================
TEST_CASE("spacetime_symbolic_doublegyre", "[spacetime_field]") {
  symbolic::doublegyre dg;
  spacetime_field stdg{dg};
  auto      dg_vel   = dg({0.3, 0.3}, 1);
  auto      stdg_vel = stdg({0.3, 0.3, 1.0});
  REQUIRE(dg_vel(0) == stdg_vel(0));
  REQUIRE(dg_vel(1) == stdg_vel(1));
  REQUIRE(stdg_vel(2) == 1);
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
