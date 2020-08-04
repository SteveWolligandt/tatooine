#include <tatooine/polynomial_line.h>
#include <tatooine/interpolation.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("polynomial_line", "[polynomial_line][constructor]") {
  polynomial_line l{polynomial{0.0, 0.0, 1.0},
                    polynomial{0.0, 1.0},
                    polynomial{0.0, 3.0},
                    polynomial{0.0, 1.0, 0.0, 10.0}};
  REQUIRE(l.degree() == 3);
  REQUIRE(l.num_dimensions() == 4);
  auto x = l(2);
  REQUIRE(x(0) == Approx(2 * 2));
  REQUIRE(x(1) == Approx(2));
  REQUIRE(x(2) == Approx(2 * 3));
  REQUIRE(x(3) == Approx(2 + 2 * 2 * 2 * 10));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
