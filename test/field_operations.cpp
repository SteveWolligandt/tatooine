#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/field_operations.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//═════════════════════════════════════════════════════════════════════════════╗
namespace tatooine::test {
//═══════════════════════════════════════════════════════════════════════════╗
TEST_CASE("field_operations_dot", "[field_operations]") {
  auto   v          = analytical::numerical::doublegyre{};
  auto   sqr_len_dg = dot(v, v);
  auto   x          = vec<double, 2>{0.1, 0.1};
  auto t = real_number{1};
  REQUIRE(dot(v(x, t), v(x, t)) == Approx(sqr_len_dg(x, t)));
}
//═══════════════════════════════════════════════════════════════════════════╡
TEST_CASE("field_operations_tensor_sum", "[field_operations]") {
  auto v = analytical::numerical::doublegyre{};
  auto dg2 = v + v;
  REQUIRE(dg2(0.3, 0.3)(0) == Approx(v(0.3, 0.3)(0) * 2));
  REQUIRE(dg2(0.3, 0.3)(1) == Approx(v(0.3, 0.3)(1) * 2));
}
//═══════════════════════════════════════════════════════════════════════════╝
}  // namespace tatooine::test
//═════════════════════════════════════════════════════════════════════════════╝
