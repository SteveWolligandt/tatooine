#include "../doublegyre.h"
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("field_operations_dot", "[field_operations]") {
  numerical::doublegyre dg;
  auto sqr_len_dg = dot(dg, dg);

  std::cout << sqr_len_dg({0, 0}, 0) << '\n';
}

TEST_CASE("field_operations_tensor_sum", "[field_operations]") {
  numerical::doublegyre dg;
  auto dg2 = dg + dg;
  REQUIRE(dg2({0.3, 0.3}, 0)(0) == Approx(dg({0.3, 0.3}, 0)(0) * 2));
  REQUIRE(dg2({0.3, 0.3}, 0)(1) == Approx(dg({0.3, 0.3}, 0)(1) * 2));
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
