#include <tatooine/static_multidim_size.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("static_multidim_size", "[static_multidim_size]") {
  using size = static_multidim_size<x_fastest, 2, 2>;
  REQUIRE(size::in_range(0,0));
  REQUIRE(size::in_range(1,0));
  REQUIRE(size::in_range(0,1));
  REQUIRE(size::in_range(1,1));
  REQUIRE_FALSE(size::in_range(2,1));
  REQUIRE_FALSE(size::in_range(-1,1));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
