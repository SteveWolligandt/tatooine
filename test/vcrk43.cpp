#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::ode::vclibs::test {
//==============================================================================
TEST_CASE("vclibs_rungekutta43",
          "[vc][rk43][rungekutta43][ode][integrator][integration]") {
  analytical::fields::numerical::doublegyre v;
  rungekutta43<double, 2> rk43;
  rk43.solve(v, vec{0.1, 0.1}, 0, 10, [](auto const &y, auto const t) {
    std::cerr << y << ", " << t << '\n';
  });
}
//==============================================================================
}  // namespace tatooine::ode::vclibs::test
//==============================================================================

