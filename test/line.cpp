#include <tatooine/line.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("line", "[line]") {
  line<double, 2> l;
  l.push_back({0,0});
  l.push_back({1,1});
  l.push_back({2,0});

  auto t0 = l.tangent(0);
  auto t1 = l.tangent(1);
  auto t2 = l.tangent(2);
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
