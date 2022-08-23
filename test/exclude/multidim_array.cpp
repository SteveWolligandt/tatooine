#include <catch2/catch_test_macros.hpp>
#include <tatooine/multidim_array.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("static_multidim_stack_array", "[multidim_array][static][stack]") {
  SECTION("x_fastest") {
    auto arr = static_multidim_array<int, x_fastest, stack, 3, 4, 2>::ones();
    REQUIRE(arr.num_dimensions() == 3);
    REQUIRE(arr.num_elements() == 3 * 4 * 2);
    REQUIRE(arr(1, 1, 1) == 1);
    REQUIRE(arr[1 + 3 + 3 * 4] == 1);
    arr(1, 1, 1) = 2;
    REQUIRE(arr(1, 1, 1) == 2);
    REQUIRE(arr[1 + 3 + 3 * 4] == 2);

    static_multidim_array<int, x_slowest, heap, 3, 4, 2> arr2;
    arr2.randu();
    arr = arr2;
    for (auto is : arr.indices()) { REQUIRE(arr(is) == arr2(is)); }
  }
  SECTION("x_slowest") {
    auto arr = static_multidim_array<int, x_slowest, stack, 3, 4, 2>::ones();
    REQUIRE(arr.num_dimensions() == 3);
    REQUIRE(arr.num_elements() == 3 * 4 * 2);
    REQUIRE(arr(1, 1, 1) == 1);
    REQUIRE(arr[1 + 2 + 2 * 4] == 1);
    arr(1, 1, 1) = 2;
    REQUIRE(arr(1, 1, 1) == 2);
    REQUIRE(arr[1 + 2 + 2 * 4] == 2);

    static_multidim_array<int, x_fastest, heap, 3, 4, 2> arr2;
    arr2.randu();
    arr = arr2;
    for (auto is : arr.indices()) { REQUIRE(arr(is) == arr2(is)); }
  }
}
//==============================================================================
TEST_CASE("static_multidim_heap_array", "[multidim_array][static][heap]") {
  SECTION("x_fastest") {
    auto arr =static_multidim_array<int, x_fastest, heap, 3, 4, 2>::ones();
    REQUIRE(arr.num_dimensions() == 3);
    REQUIRE(arr.num_elements() == 3 * 4 * 2);
    REQUIRE(arr(1, 1, 1) == 1);
    REQUIRE(arr[1 + 3 + 3 * 4] == 1);
    arr(1, 1, 1) = 2;
    REQUIRE(arr(1, 1, 1) == 2);
    REQUIRE(arr[1 + 3 + 3 * 4] == 2);

    static_multidim_array<int, x_slowest, stack, 3, 4, 2> arr2;
    arr2.randu();
    arr = arr2;
    for (auto is : arr.indices()) { REQUIRE(arr(is) == arr2(is)); }
  }
  SECTION("x_slowest") {
    auto arr = static_multidim_array<int, x_slowest, heap, 3, 4, 2>::ones();
    REQUIRE(arr.num_dimensions() == 3);
    REQUIRE(arr.num_elements() == 3 * 4 * 2);
    REQUIRE(arr(1, 1, 1) == 1);
    REQUIRE(arr[1 + 2 + 2 * 4] == 1);
    arr(1, 1, 1) = 2;
    REQUIRE(arr(1, 1, 1) == 2);
    REQUIRE(arr[1 + 2 + 2 * 4] == 2);

    const auto arr2 =
        static_multidim_array<int, x_fastest, stack, 3, 4, 2>::randu(0, 100);
    arr = arr2;
    for (auto is : arr.indices()) { REQUIRE(arr(is) == arr2(is)); }
  }
}
//==============================================================================
TEST_CASE("dynamic_multidim_array", "[multidim_array][dynamic]") {
  SECTION("x_fastest") {
    auto arr = dynamic_multidim_array<int, x_fastest>::ones(3, 4, 2);
    REQUIRE(arr.num_dimensions() == 3);
    REQUIRE(arr.num_elements() == 3 * 4 * 2);
    REQUIRE(arr(1, 1, 1) == 1);
    REQUIRE(arr[1 + 3 + 3 * 4] == 1);
    arr(1, 1, 1) = 2;
    REQUIRE(arr(1, 1, 1) == 2);
    REQUIRE(arr[1 + 3 + 3 * 4] == 2);

    arr.resize(5, 6);
    REQUIRE(arr.num_dimensions() == 2);
    REQUIRE(arr.num_elements() == 5 * 6);
    arr(2, 4) = 0;
    REQUIRE(arr[2 + 4 * 5] == 0);

    const auto arr2 = dynamic_multidim_array<int, x_slowest>::randu(
        0, 99, std::vector<size_t>{2, 2, 2, 2});
    arr = arr2;
    REQUIRE(arr.num_dimensions() == 4);
    REQUIRE(arr.num_elements() == 2 * 2 * 2 * 2);

    for (auto is : arr.indices()) { REQUIRE(arr(is) == arr2(is)); }
  }
  SECTION("x_slowest") {
    auto arr = dynamic_multidim_array<int, x_slowest>::ones(3, 4, 2);
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

    const auto arr2 = dynamic_multidim_array<int, x_fastest>::randu(
        0, 100, std::vector<size_t>{3, 4, 2});
    arr = arr2;
    for (auto is : arr.indices()) { REQUIRE(arr(is) == arr2(is)); }
  }
}
//==============================================================================
TEST_CASE("static_multidim_array_interpolate",
          "[multidim_array][interpolate][static]") {
  using array_t = static_multidim_array<double, x_slowest, stack, 3, 4, 2>;
  auto arr1     = array_t::randu();
  auto arr2     = array_t::randu();
  auto arr1_2   = interpolate(arr1, arr2, 0.2);

  for (auto is : arr1.indices()) {
    REQUIRE(arr1_2(is) == Approx(arr1(is) * 0.8 + arr2(is) * 0.2).margin(1e-9));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
