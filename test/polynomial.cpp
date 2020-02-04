#include <tatooine/polynomial.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("polynomial_eval", "[polynomial][evaluate]") {
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
TEST_CASE("polynomial_diff", "[polynomial][diff]") {
  SECTION("1 -> 0") {
    polynomial<double, 0> f{1.0};
    auto dfdx = diff(f);
    CAPTURE(f, dfdx);
    REQUIRE(dfdx.num_degrees() == 0);
    REQUIRE(dfdx.c(0) == Approx(0).margin(1e-10));
  }
  SECTION("x -> 1") {
    polynomial f{0.0, 1.0};
    auto dfdx = diff(f);
    CAPTURE(f, dfdx);
    REQUIRE(dfdx.num_degrees() == 0);
    REQUIRE(dfdx.c(0) == Approx(1).margin(1e-10));
  }
  SECTION("x^2 -> 2x") {
    polynomial f{0.0, 0.0, 1.0};
    auto dfdx = diff(f);
    CAPTURE(f, dfdx);
    REQUIRE(dfdx.num_degrees() == 1);
    REQUIRE(dfdx.c(0) == Approx(0).margin(1e-10));
    REQUIRE(dfdx.c(1) == Approx(2).margin(1e-10));
  }
  SECTION("x + 2x^2 -> 1 + 4x") {
    polynomial f{0.0, 1.0, 2.0};
    auto dfdx = diff(f);
    CAPTURE(f, dfdx);
    REQUIRE(dfdx.num_degrees() == 1);
    REQUIRE(dfdx.c(0) == Approx(1).margin(1e-10));
    REQUIRE(dfdx.c(1) == Approx(4).margin(1e-10));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
