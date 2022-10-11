#include <tatooine/linspace.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <catch2/matchers/catch_matchers_all.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace Catch;
TEST_CASE("linspace", "[linspace][range]") {
  auto lin = linspace{0.0, 1.0, 11};
  REQUIRE(lin[0] == 0.0);
  REQUIRE(lin[1] == Approx(0.1));
  REQUIRE(lin[10] == 1.0);
  REQUIRE(*begin(lin) == 0.0);
  REQUIRE(*next(begin(lin)) == 0.1);
  auto it = begin(lin);
  it.increment();
  auto as_vector = std::vector<decltype(lin)::value_type>{};
  std::ranges::copy(lin, std::back_inserter(as_vector));
  REQUIRE_THAT(as_vector,
               Catch::Matchers::Approx(std::vector{0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                                   0.6, 0.7, 0.8, 0.9, 1.0}));

  auto lin_it = begin(lin);
  auto vec_it = begin(as_vector);

  REQUIRE(distance(lin_it, next(lin_it)) == distance(vec_it, next(vec_it)));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
