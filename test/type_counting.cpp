#include <tatooine/count_types.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("count_types_get_count", "[count_types][get_count]") {
  using counter =
      type_counter_impl<type_number_pair<int, 1>, type_number_pair<float, 2>>;
  REQUIRE(get_count<counter, int> == 1);
  REQUIRE(get_count<counter, float> == 2);
}
//==============================================================================
TEST_CASE("count_types_insert_type",
          "[count_types][insert_types_into_counter]") {
  using base_counter = type_counter_impl<type_number_pair<int, 0>>;
  using counter      = insert_types_into_counter<base_counter, int>;
  REQUIRE(get_count<counter, int> == 1);
}
//==============================================================================
TEST_CASE("count_types_insert_types",
          "[count_types][insert_types_into_counter]") {
  using base_counter =
      type_counter_impl<type_number_pair<int, 0>, type_number_pair<float, 0>>;
  using counter = insert_types_into_counter<base_counter, int, float, float>;
  REQUIRE(get_count<counter, int> == 1);
  REQUIRE(get_count<counter, float> == 2);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
