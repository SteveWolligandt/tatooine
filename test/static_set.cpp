#include <tatooine/type_set.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("type_set_contains", "[type_set][type][contains]") {
  using Set = type_set<int, float, double>;
  REQUIRE(type_list_contains<Set, int>);
  REQUIRE(type_list_contains<Set, float>);
  REQUIRE(type_list_contains<Set, double>);
  REQUIRE_FALSE(type_list_contains<Set, unsigned int>);
}
//==============================================================================
TEST_CASE("type_set_size", "[type_set][type][size]") {
  using Set = type_set_impl<int, float, double>;
  REQUIRE(type_list_size<Set> == 3);
}
//==============================================================================
TEST_CASE("type_set_constructor", "[type_set][type][constructor]") {
  using Set = type_set<int, float, int, float, double, double>;
  REQUIRE(type_list_size<Set> == 3);
  REQUIRE(type_list_contains<Set, int>);
  REQUIRE(type_list_contains<Set, float>);
  REQUIRE(type_list_contains<Set, double>);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
