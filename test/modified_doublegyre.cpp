#include <tatooine/analytical/numerical/modified_doublegyre.h>
#include <tatooine/ode/vclibs/rungekutta43.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("modified_doublegyre_lcs_creation",
          "[modified_doublegyre][lcs][creation]") {
  analytical::numerical::modified_doublegyre v;
  auto ht = v.hyperbolic_trajectory();
  line<real_type, 2> lcs;
  real_type const t0 = 0;
  real_type const tau = 20;
  real_type const step_width = 0.1;
  real_type const mu = 1e-6;
  ode::vclibs::rungekutta43<real_type, 2> ode;

  lcs.push_back(ht(t0));
  real_type t = t0;
  while (t < t0 + tau) {
    t += step_width;
    vec2 y;
    auto x0 = ht(t);
    x0(1) += mu;
    ode.solve(v, x0, t, t0 - t,
              [&y](auto const& x, auto const /*t*/) { y = x; });
    lcs.push_back(y);
  }
  lcs.write_vtk("lcs_mdg.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
