#include <tatooine/static_set.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("static_set_includes", "[static_set][type][includes]") {
  using Set = static_type_set_impl<int, float, double>;
  REQUIRE(static_set_includes<Set, int>);
  REQUIRE(static_set_includes<Set, float>);
  REQUIRE(static_set_includes<Set, double>);
  REQUIRE_FALSE(static_set_includes<Set, unsigned int>);
}
//==============================================================================
TEST_CASE("static_set_size", "[static_set][type][size]") {
  using Set = static_type_set_impl<int, float, double>;
  REQUIRE(static_set_size<Set> == 3);
}
//==============================================================================
TEST_CASE("static_set_constructor", "[static_set][type][constructor]") {
  using Set = static_type_set<int, float, int, float, double, double>;
  REQUIRE(static_set_size<Set> == 3);
  REQUIRE(static_set_includes<Set, int>);
  REQUIRE(static_set_includes<Set, float>);
  REQUIRE(static_set_includes<Set, double>);
}
//==============================================================================
} // namespace tatooine::test
//==============================================================================
