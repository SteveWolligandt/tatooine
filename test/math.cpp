#include <tatooine/math.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test{
TEST_CASE("math_minmax", "[math][max]") {
  REQUIRE(max(2,3) == 3);
  REQUIRE(min(2,3) == 2);
  REQUIRE(max(1,2,3) == 3);
  REQUIRE(min(1,2,3) == 1);
}
}
