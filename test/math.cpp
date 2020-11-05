#include <tatooine/math.h>

#include <catch2/catch.hpp>
//==============================================================================
TEST_CASE("math_max", "[math][max]") {
  SECTION("same types") {
    SECTION("all non-const") {
      int i0 = 2, i1 = 3, i2 = 1;
      REQUIRE(tatooine::max(i0, i1, i2) == i1);
      REQUIRE(std::is_same_v<decltype(tatooine::max(i0, i1, i2)), int &>);
    }
    SECTION("mixed const") {
      int       i0 = 2, i1 = 3;
      int const i2 = 1;
      REQUIRE(tatooine::max(i0, i1, i2) == i1);
      REQUIRE(std::is_same_v<decltype(tatooine::max(i0, i1, i2)), int const &>);
    }
  }
  SECTION("different types") {
    int    i = 2;
    float  f = 3;
    double d = 1;
    REQUIRE(tatooine::max(i, f, d) == f);
    REQUIRE(std::is_same_v<decltype(tatooine::max(i, f, d)), double>);
  }
}
