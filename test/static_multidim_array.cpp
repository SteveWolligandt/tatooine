#include <tatooine/static_multidim_array.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("static_multidim_array", "[static_multidim_array]") {
  auto arr = static_multidim_array<int, x_fastest, tag::stack, 2, 2>{};
  arr(0,0);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
