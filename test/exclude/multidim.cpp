#include <catch2/catch_test_macros.hpp>
#include <tatooine/multidim.h>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("static_multidim", "[multidim][static]") {
  static_multidim m{2, 2, 3};
  REQUIRE(m[0].first == 0); REQUIRE(m[0].second == 2);
  REQUIRE(m[1].first == 0); REQUIRE(m[1].second == 2);
  REQUIRE(m[2].first == 0); REQUIRE(m[2].second == 3);

  auto it = m.begin();
  REQUIRE((*it)[0] == 0);
  REQUIRE((*it)[1] == 0);
  REQUIRE((*it)[2] == 0);

  ++it;
  REQUIRE((*it)[0] == 1);
  REQUIRE((*it)[1] == 0);
  REQUIRE((*it)[2] == 0);

  ++it;
  REQUIRE((*it)[0] == 0);
  REQUIRE((*it)[1] == 1);
  REQUIRE((*it)[2] == 0);

  ++it;
  REQUIRE((*it)[0] == 1);
  REQUIRE((*it)[1] == 1);
  REQUIRE((*it)[2] == 0);

  ++it;
  REQUIRE((*it)[0] == 0);
  REQUIRE((*it)[1] == 0);
  REQUIRE((*it)[2] == 1);
  
}
//==============================================================================
TEST_CASE("dynamic_multidim", "[multidim][dynamic]") {}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
