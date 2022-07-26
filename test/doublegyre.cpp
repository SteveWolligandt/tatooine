#include <tatooine/analytical/numerical/doublegyre.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::analytical::test {
//==============================================================================
TEST_CASE("doublegyre", "[dg][doublegyre]") {
  numerical::doublegyre v;
  REQUIRE(approx_equal(v(0, 0, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(1, 0, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(2, 0, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(0, 1, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(1, 1, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(2, 1, 0), vec2::zeros()));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
