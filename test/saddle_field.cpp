#include <tatooine/analytical/numerical/saddle.h>
#include <tatooine/numerical_flowmap.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("saddle_field_flowmap","[saddle][flowmap]") {
  analytical::numerical::saddle v;
  [[maybe_unused]] auto fma   = flowmap(v, tag::analytical);
  //[[maybe_unused]] auto fmaga = diff(fma, tag::analytical);
  //[[maybe_unused]] auto fmagc = diff(fma, tag::central, 1e-7);
  [[maybe_unused]] auto fmn   = flowmap(v, tag::numerical);
  //[[maybe_unused]] auto fmngc = diff(fmn, tag::central, 1e-7);
  REQUIRE(
      same_as<decltype(fma), analytical::numerical::saddle_flowmap<double>>);
  REQUIRE(
      same_as<decltype(fmn),
              numerical_flowmap<analytical::numerical::saddle<double> const&,
                                ode::boost::rungekuttafehlberg78,
                                interpolation::cubic>>);
}
//==============================================================================
}  // namespace tatooine::analytical::test
//==============================================================================
