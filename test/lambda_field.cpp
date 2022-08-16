#include <catch2/catch.hpp>
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("lambda_field_circle") {
  auto constexpr v   = make_field<2>([](auto const& x, auto const& /*t*/) {
    return vec{-x.y(), x.x()};
  });
  constexpr auto v00 = v(0, 0);
  constexpr auto v12 = v(1, 2);
  REQUIRE(v00(0) == 0);
  REQUIRE(v12(0) == -2);
  REQUIRE(v12(1) == 1);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
