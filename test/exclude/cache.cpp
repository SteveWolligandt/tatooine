#include <tatooine/cache.h>
#include <tatooine/tensor.h>
#include <catch2/catch_test_macros.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("cache", "[cache]") {
  cache<std::pair<double, vec<double, 3>>, int> c;
  auto [it_x, x_inserted] = c.try_emplace({0.0, vec{0.1, 0.1, 0.0}}, 1);
  auto [it_y, y_inserted] = c.try_emplace({0.0, vec{0.2, 0.2, 0.0}}, 2);

  auto& x = it_x->second;
  auto& y = it_y->second;

  REQUIRE(x_inserted);
  REQUIRE(y_inserted);

  REQUIRE(x == 1);
  REQUIRE(y == 2);
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
