#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/flowmap_gradient_central_differences.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::analytical::fields::numerical::test {
//==============================================================================
TEST_CASE("saddle_field_flowmap","[saddle][flowmap]") {
  saddle v;
  auto fma = flowmap(v, tag::analytical);
  auto fmaga = diff(fma, tag::analytical);
  auto fmagc = diff(fma, tag::central);
  auto fmn = flowmap(v, tag::numerical);
  auto fmngc = diff(fmn, tag::central);
}
//==============================================================================
}  // namespace tatooine::analytical::fields::test
//==============================================================================
