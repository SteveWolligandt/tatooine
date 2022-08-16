#include <tatooine/ode/vclibs/rungekutta43.h>
#include <tatooine/analytical/numerical/doublegyre.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("ode_vclibs_rk43", "[ode][rk43][vclibs][rungekutta][rungekutta43]") {
  using namespace ode::vclibs;
  auto       solver = rungekutta43<real_number, 2>{};
  auto const v      = analytical::numerical::doublegyre{};
  auto       last_t = real_number{};
  auto const stop_t = real_number{};
  solver.solve(
      [&](auto const& x, auto const t) -> maybe_vec_t<real_number, 2> {
        auto sample = v(x, t);
        if (sample.isnan()) {
          return out_of_domain;
        }
        if (t >= stop_t) {
          return failed;
        }
        last_t = t;
        return sample;
      },
      vec{0.1, 0.1}, 0, 10, [](auto const& /*x*/, auto const /*t*/) {});
  CAPTURE(last_t, stop_t, std::abs(last_t - stop_t));
  REQUIRE(last_t <= stop_t);
}
//==============================================================================
}
//==============================================================================
