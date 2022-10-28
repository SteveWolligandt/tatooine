#include <tatooine/dynamic_multidim_array.h>
#include <tatooine/tensor.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("dynamic_multidim_array", "[dynamic_multidim_array]") {
  auto arr = dynamic_multidim_array<int>{2, 3, 4};
  REQUIRE(arr.num_dimensions() == 3);
  REQUIRE(arr.size(0) == 2);
  REQUIRE(arr.size(1) == 3);
  REQUIRE(arr.size(2) == 4);
  arr(1, 1, 1) = 5;
  arr.resize(3, 4, 5);
  REQUIRE(arr.num_dimensions() == 3);
  REQUIRE(arr.size(0) == 3);
  REQUIRE(arr.size(1) == 4);
  REQUIRE(arr.size(2) == 5);
  REQUIRE(arr(1, 1, 1) == 5);
  arr.resize(2, 2, 2);
  REQUIRE(arr.num_dimensions() == 3);
  REQUIRE(arr.size(0) == 2);
  REQUIRE(arr.size(1) == 2);
  REQUIRE(arr.size(2) == 2);
  REQUIRE(arr(1, 1, 1) == 5);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
