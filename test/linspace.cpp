#include <tatooine/linspace.h>
#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("linspace", "[linspace]") {
  auto ts = linspace{0.0, 1.0, 11};
  REQUIRE(ts.front() == 0.0);
  REQUIRE(ts.back() == 1.0);
  REQUIRE(ts.size() == 11);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
