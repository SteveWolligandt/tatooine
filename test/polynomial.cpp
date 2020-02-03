#include <tatooine/polynomial.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("line_push_back", "[line][push_back]") {
  SECTION("x^2") {
    polynomial f{0.0, 0.0, 1.0};
    REQUIRE(f(2) == Approx(2 * 2));
    REQUIRE(f(3) == Approx(3 * 3));
  }
  SECTION("x + 2*x^2") {
    polynomial f{0.0, 1.0, 2.0};
    REQUIRE(f(2) == Approx(2 + 2 * 2 * 2));
    REQUIRE(f(3) == Approx(3 + 2 * 3 * 3));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
