#include <tatooine/vec.h>
#include <tatooine/mat.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("lapack", "[lapack]") {
  auto A    = mat2{{1, 2},
                   {2, 3}};
  auto b    = vec2 {3, 7};
  auto ipiv = vec2i64{};
  [[maybe_unused]] auto res  = lapack::gesv(A, b, ipiv);
  REQUIRE(b(0) == 5);
  REQUIRE(b(1) == -1);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
