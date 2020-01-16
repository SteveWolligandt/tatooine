#include <tatooine/doublegyre.h>
#include <tatooine/diff.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("symbolic_field_acceleration", "[symbolic_field][doublegyre][dg]") {
  symbolic::doublegyre dg;
  auto jdg = diff(dg);
  auto dg_acc = jdg * dg;
  auto jdg_acc = diff(dg_acc);
  auto dg_jerk = jdg_acc * dg;
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
