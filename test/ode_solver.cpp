#include <tatooine/ode/vclibs/rungekutta43.h>
#include <tatooine/analytical/numerical/doublegyre.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("ode_vclibs_rk43", "[ode][rk43][vclibs][rungekutta][rungekutta43]") {
  using namespace ode::vclibs;
  rungekutta43<double, 2>            solver;
  analytical::numerical::doublegyre const v;
  double                                          last_t = 0;
  double const                                    stop_t = 2;
  solver.solve(
      [&](auto const& x, auto const t) -> maybe_vec_t<double, 2> {
        if (!v.in_domain(x, t)) { return out_of_domain; }
        if (t >= stop_t) { return failed; }
        last_t = t;
        return v(x, t);
      },
      vec{0.1, 0.1}, 0, 10,
      [](auto const& /*x*/, auto const /*t*/) {});
  CAPTURE(last_t, stop_t, std::abs(last_t - stop_t));
  REQUIRE(last_t <= stop_t);
}
//==============================================================================
}
//==============================================================================
