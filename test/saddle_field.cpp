#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/flowmap_gradient_central_differences.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("saddle_field_flowmap","[saddle][flowmap]") {
  analytical::fields::numerical::saddle v;
  [[maybe_unused]] auto fma   = flowmap(v, tag::analytical);
  [[maybe_unused]] auto fmaga = diff(fma, tag::analytical);
  [[maybe_unused]] auto fmagc = diff(fma, tag::central, 1e-7);
  [[maybe_unused]] auto fmn   = flowmap(v, tag::numerical);
  [[maybe_unused]] auto fmngc = diff(fmn, tag::central, 1e-7);
  REQUIRE(
      std::is_same_v<decltype(fma),
                     analytical::fields::numerical::saddle_flowmap<double>>);
  REQUIRE(
      std::is_same_v<decltype(fmn),
                     numerical_flowmap<
                         analytical::fields::numerical::saddle<double>,
                         ode::vclibs::rungekutta43, interpolation::cubic>>);
}
//==============================================================================
}  // namespace tatooine::analytical::fields::test
//==============================================================================
