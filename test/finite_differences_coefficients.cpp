#include <catch2/catch.hpp>
#include <tatooine/finite_differences_coefficients.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("finite_differences_coefficients") {
  auto coeffs = finite_differences_coefficients(4, -2.0, -1.0, 0.0, 1.0, 2.0);
  CAPTURE(coeffs);
  REQUIRE(coeffs.num_components() == 5);
  REQUIRE(coeffs(0) == Approx(1));
  REQUIRE(coeffs(1) == Approx(-4));
  REQUIRE(coeffs(2) == Approx(6));
  REQUIRE(coeffs(3) == Approx(-4));
  REQUIRE(coeffs(4) == Approx(1));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
