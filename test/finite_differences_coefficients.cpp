#include <catch2/catch.hpp>
#include <tatooine/finite_differences_coefficients.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("finite_differences_coefficients_central_1_3") {
  auto coeffs = finite_differences_coefficients(1, -1.0, 0.0, 1.0);
  CAPTURE(coeffs);
  REQUIRE(coeffs.num_components() == 3);
  REQUIRE(coeffs(0) == Approx(-0.5));
  REQUIRE(coeffs(1) == Approx(0));
  REQUIRE(coeffs(2) == Approx(0.5));
}
//==============================================================================
TEST_CASE("finite_differences_coefficients_central_2_3") {
  auto coeffs = finite_differences_coefficients(2, -1.0, 0.0, 1.0);
  CAPTURE(coeffs);
  REQUIRE(coeffs.num_components() == 3);
  REQUIRE(coeffs(0) == Approx(1));
  REQUIRE(coeffs(1) == Approx(-2));
  REQUIRE(coeffs(2) == Approx(1));
}
//==============================================================================
TEST_CASE("finite_differences_central_4_5") {
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
TEST_CASE("finite_differences_central_1_5") {
  auto coeffs = finite_differences_coefficients(1, -2.0, -1.0, 0.0, 1.0, 2.0);
  CAPTURE(coeffs);
  REQUIRE(coeffs.num_components() == 5);
  REQUIRE(coeffs(0) == Approx(1.0 / 12.0));
  REQUIRE(coeffs(1) == Approx(-8.0 / 12.0));
  REQUIRE(coeffs(2) == Approx(0).margin(1e-7));
  REQUIRE(coeffs(3) == Approx(8.0 / 12.0));
  REQUIRE(coeffs(4) == Approx(-1.0 / 12.0));
}
//==============================================================================
TEST_CASE("finite_differences_1_4") {
  auto coeffs = finite_differences_coefficients(1, -2.1, -1.5, 0.0, 3.8);
  CAPTURE(coeffs);
  REQUIRE(coeffs.num_components() == 4);
  REQUIRE(coeffs(0) == Approx(956650.0 / 1247673.0));
  REQUIRE(coeffs(1) == Approx(-2087302.0 / 1247673.0));
  REQUIRE(coeffs(2) == Approx(1097577.0 / 1247673.0));
  REQUIRE(coeffs(3) == Approx(33075.0 / 1247673.0));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
