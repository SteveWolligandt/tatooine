#include <catch2/catch.hpp>
#include <tatooine/multidim_array.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("static_multidim_array", "[multidim_array][static]") {
  SECTION("x_fastest") {
    static_multidim_array<int, x_fastest, 3, 4, 2> arr{1};
    REQUIRE(arr.num_dimensions() == 3);
    REQUIRE(arr.num_elements() == 3 * 4 * 2);
    REQUIRE(arr(1, 1, 1) == 1);
    REQUIRE(arr[1 + 3 + 3 * 4] == 1);
    arr(1, 1, 1) = 2;
    REQUIRE(arr(1, 1, 1) == 2);
    REQUIRE(arr[1 + 3 + 3 * 4] == 2);

    static_multidim_array<int, x_slowest, 3, 4, 2> arr2;
    arr2.randu();
    arr = arr2;
    for (auto is : arr.indices()) { REQUIRE(arr(is) == arr2(is)); }
  }
  SECTION("x_slowest") {
    static_multidim_array<int, x_slowest, 3, 4, 2> arr{1};
    REQUIRE(arr.num_dimensions() == 3);
    REQUIRE(arr.num_elements() == 3 * 4 * 2);
    REQUIRE(arr(1, 1, 1) == 1);
    REQUIRE(arr[1 + 2 + 2 * 4] == 1);
    arr(1, 1, 1) = 2;
    REQUIRE(arr(1, 1, 1) == 2);
    REQUIRE(arr[1 + 2 + 2 * 4] == 2);

    static_multidim_array<int, x_fastest, 3, 4, 2> arr2;
    arr2.randu();
    arr = arr2;
    for (auto is : arr.indices()) { REQUIRE(arr(is) == arr2(is)); }
  }
}
//==============================================================================
TEST_CASE("dynamic_multidim_array", "[multidim_array][dynamic]") {
  SECTION("x_fastest") {
    dynamic_multidim_array<int, x_fastest> arr{std::vector<size_t>{3, 4, 2}, 1};
    REQUIRE(arr.num_dimensions() == 3);
    REQUIRE(arr.num_elements() == 3 * 4 * 2);
    REQUIRE(arr(1, 1, 1) == 1);
    REQUIRE(arr[1 + 3 + 3 * 4] == 1);
    arr(1, 1, 1) = 2;
    REQUIRE(arr(1, 1, 1) == 2);
    REQUIRE(arr[1 + 3 + 3 * 4] == 2);

    arr.resize(5,6);
    REQUIRE(arr.num_dimensions() == 2);
    REQUIRE(arr.num_elements() == 5 * 6);
    arr(2, 4) = 0;
    REQUIRE(arr[2 + 4 * 5] == 0);
  }
  SECTION("x_slowest") {
    dynamic_multidim_array<int, x_slowest> arr{std::vector<size_t>{3, 4, 2}, 1};
    REQUIRE(arr.num_dimensions() == 3);
    REQUIRE(arr.num_elements() == 3 * 4 * 2);
    REQUIRE(arr(1, 1, 1) == 1);
    REQUIRE(arr[1 + 2 + 2 * 4] == 1);
    arr(1, 1, 1) = 2;
    REQUIRE(arr(1, 1, 1) == 2);
    REQUIRE(arr[1 + 2 + 2 * 4] == 2);

    arr.resize(5,6);
    REQUIRE(arr.num_dimensions() == 2);
    REQUIRE(arr.num_elements() == 5 * 6);
    arr(2, 4) = 0;
    REQUIRE(arr[4 + 6 * 2] == 0);
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
