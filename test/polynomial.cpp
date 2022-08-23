#include <tatooine/polynomial.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
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
    REQUIRE(dfdx.degree() == 0);
    REQUIRE(dfdx.c(0) == Approx(0).margin(1e-10));
  }
  SECTION("x -> 1") {
    polynomial f{0.0, 1.0};
    auto dfdx = diff(f);
    CAPTURE(f, dfdx);
    REQUIRE(dfdx.degree() == 0);
    REQUIRE(dfdx.c(0) == Approx(1).margin(1e-10));
  }
  SECTION("x^2 -> 2x") {
    polynomial f{0.0, 0.0, 1.0};
    auto dfdx = diff(f);
    CAPTURE(f, dfdx);
    REQUIRE(dfdx.degree() == 1);
    REQUIRE(dfdx.c(0) == Approx(0).margin(1e-10));
    REQUIRE(dfdx.c(1) == Approx(2).margin(1e-10));
  }
  SECTION("x + 2x^2 -> 1 + 4x") {
    polynomial f{0.0, 1.0, 2.0};
    auto dfdx = diff(f);
    CAPTURE(f, dfdx);
    REQUIRE(dfdx.degree() == 1);
    REQUIRE(dfdx.c(0) == Approx(1).margin(1e-10));
    REQUIRE(dfdx.c(1) == Approx(4).margin(1e-10));
  }
}
//==============================================================================
TEST_CASE("polynomial_solve_quadratic", "[polynomial][solve][quadratic]") {
  SECTION("f(x) = 1") {
    polynomial f{1.0, 0.0, 0.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 0);
  }
  SECTION("f(x) = 1+x") {
    polynomial f{1.0, 1.0, 0.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 1);
    REQUIRE(solutions[0] == Approx(-1).margin(1e-10));
  }
  SECTION("f(x) = x^2") {
    polynomial f{0.0, 0.0, 1.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 1);
    REQUIRE(solutions[0] == Approx(0).margin(1e-10));
  }
  SECTION("f(x) = -1 + x^2") {
    polynomial f{-1.0, 0.0, 1.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 2);
    REQUIRE(solutions[0] == Approx(-1).margin(1e-10));
    REQUIRE(solutions[1] == Approx(1).margin(1e-10));
  }
  SECTION("f(x) = x + 2x^2") {
    polynomial f{0.0, 1.0, 2.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 2);
    REQUIRE(solutions[0] == Approx(0).margin(1e-10));
    REQUIRE(solutions[1] == Approx(-0.5).margin(1e-10));
  }
}
//==============================================================================
TEST_CASE("polynomial_solve_cubic", "[polynomial][solve][cubic]") {
  SECTION("f(x) = x^3") {
    polynomial f{0.0, 0.0, 0.0, 1.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 1);
    REQUIRE(solutions[0] == Approx(0).margin(1e-10));
  }
  SECTION("f(x) = x^3") {
    polynomial f{0.0, 0.0, 1.0, 1.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 2);
    REQUIRE(solutions[0] == Approx(-1).margin(1e-10));
    REQUIRE(solutions[1] == Approx(0).margin(1e-10));
  }
  SECTION("f(x) = 1 - 2x^2 + x^3") {
    polynomial f{1.0, 0.0, -2.0, 1.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 3);
    REQUIRE(solutions[0] == Approx(-0.61803).epsilon(1e-4));
    REQUIRE(solutions[1] == Approx(1).epsilon(1e-10));
    REQUIRE(solutions[2] == Approx(1.61803).epsilon(1e-4));
  }
  SECTION("f(x) = 3.86107 + 2.56786x + 0.513068x^2 + 0.0276684x^3") {
    polynomial f{3.86107, 2.56786, 0.513068, 0.0276684};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 3);
    REQUIRE(solutions[0] == Approx(-11.55859).epsilon(1e-4));
    REQUIRE(solutions[1] == Approx(-3.8446).epsilon(1e-4));
    REQUIRE(solutions[2] == Approx(-3.14027).epsilon(1e-4));
  }
  SECTION("f(x) = -2.25932 + -0.976557*x - 0.112494*x*x -0.00132014*x*x*x") {
    polynomial f{-2.25932, -0.976557, - 0.112494, -0.00132014};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 1);
    REQUIRE(solutions[0] == Approx(-75.7459461).epsilon(1e-4));
  }
}
//==============================================================================
TEST_CASE("polynomial_solve_quartic", "[polynomial][solve][quartic]") {
  SECTION("f(x) = x^4") {
    polynomial f{0.0, 0.0, 0.0, 0.0, 1.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 1);
    REQUIRE(solutions[0] == Approx(0).margin(1e-10));
  }
  SECTION("f(x) = x^3 + x^4") {
    polynomial f{0.0, 0.0, 0.0, 1.0, 1.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 2);
    REQUIRE(solutions[0] == Approx(-1).margin(1e-10));
    REQUIRE(solutions[1] == Approx(0).margin(1e-10));
  }
  SECTION("f(x) = -5 + 30*x^2 - x^4") {
    polynomial f{-5.0, 0.0, 30.0, 0.0, -1.0};
    auto solutions = solve(f);
    REQUIRE(size(solutions) == 4);
    REQUIRE(solutions[0] == Approx(-5.46190415).margin(1e-10));
    REQUIRE(solutions[1] == Approx(-0.409393502).margin(1e-10));
    REQUIRE(solutions[2] == Approx(0.409393502).margin(1e-10));
    REQUIRE(solutions[3] == Approx(5.46190415).margin(1e-10));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
