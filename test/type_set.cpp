#include <tatooine/type_set.h>
#include<catch2/catch.hpp>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("type_set", "[type_set]") {
  using set = type_set<int, int, float, float, double, double>;
  REQUIRE(set::size == 3);
  using set2 = set::insert<char>;
  REQUIRE(type_list_size<set2> == 4);
  using set3 = set::insert<int>;
  REQUIRE(type_list_size<set3> == 3);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
