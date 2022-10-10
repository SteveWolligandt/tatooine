#include <tatooine/linspace.h>

#include <catch2/catch_test_macros.hpp>
#include <vector>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("linspace", "[linspace][range]") {
  auto ts = linspace{0.0, 1.0, 11};
  auto it = begin(ts);
  it.increment();
  it.foo();
  //SECTION("topology") {
  //  REQUIRE(ts.front() == 0.0);
  //  REQUIRE(ts.back() == 1.0);
  //  REQUIRE(ts.size() == 11);
  //}
  //SECTION("ranges") {
  //  SECTION("range based for loop") {
  //    auto i = std::size_t{};
  //    for (auto const t : ts) {
  //      REQUIRE(t == ts[i++]);
  //    }
  //  }
  //  //SECTION("algorithms") {
  //  //  auto vector = std::vector<double>{};
  //  //  std::ranges::copy(ts, std::back_inserter(vector));
  //  //}
  //}
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
