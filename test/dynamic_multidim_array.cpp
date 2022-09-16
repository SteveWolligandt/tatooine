#include <tatooine/dynamic_multidim_array.h>
#include <tatooine/tensor.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("dynamic_multidim_array1", "[dynamic_multidim_array]") {
  auto arr = tensor<double>{1,2,3};
  arr(0, 0) = 1;
  //arr(1, 0) = 2;
  //arr(1, 1) = 3;
  //arr(0, 1) = 4;
  //REQUIRE(arr(0, 0) == 1);
  //REQUIRE(arr(1, 0) == 2);
  //REQUIRE(arr(1, 1) == 3);
  //REQUIRE(arr(0, 1) == 4);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
