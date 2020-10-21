#include <tatooine/preprocessor.h>

//#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("preprocessor_num_args", "[preprocessor][num_args]") {
  REQUIRE((TATOOINE_PP_NUM_ARGS(1, 2, 3)) == 3);
}
//==============================================================================
TEST_CASE("preprocessor_map", "[preprocessor][map]") {
  auto f[](auto x){return x};
  auto g[](auto x, auto y){return x+y};
  REQUIRE(TATOOINE_PP_MAP(f, 1) == 1);
  REQUIRE(TATOOINE_PP_MAP2(g, 1, 2) == 3);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
