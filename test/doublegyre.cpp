#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::analytical::fields::test {
//==============================================================================
TEST_CASE("doublegyre", "[dg][doublegyre]") {
  numerical::doublegyre v;
  REQUIRE(approx_equal(v(vec2{0, 0}, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(vec2{1, 0}, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(vec2{2, 0}, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(vec2{0, 1}, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(vec2{1, 1}, 0), vec2::zeros()));
  REQUIRE(approx_equal(v(vec2{2, 1}, 0), vec2::zeros()));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
