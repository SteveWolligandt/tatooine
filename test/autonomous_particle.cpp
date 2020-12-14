#include <tatooine/analytical/fields/numerical/autonomous_particles_test.h>
#include <tatooine/analytical/fields/numerical/center.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/concepts.h>
#include <tatooine/spacetime_vectorfield.h>
#include <tatooine/vtk_legacy.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace analytical::fields::numerical;
//==============================================================================
TEST_CASE("autonomous_particle_doublegyre",
          "[autonomous_particle][dg][2d][2D][doublegyre][backward_integration]["
          "single]") {
   //doublegyre v;
   //v.set_infinite_domain(true);
   //vec2         x0{1.0, 0.5};
   //real_t const tau = 1;
  saddle       v;
  vec2         x0{0, 0};
  real_t const tau      = 5;

  real_t const t0       = 0;
  real_t const tau_step = 0.1;
  real_t const radius   = 0.0001;

  real_t const eps = 1e-2;

  autonomous_particle particle{v, x0, t0, radius};
  auto                advected_particles = particle.step_until_split(
      tau_step, tau, 4, true, std::array<real_t, 1>{real_t(1) / 2});
  REQUIRE(size(advected_particles) > 1);
  
  SECTION("forward integration"){
  for (auto const& p : advected_particles) {
    auto const numerical_integration = p.phi()(p.x0(), t0, tau);
    auto const dist                  = distance(numerical_integration, p.x1());
    CAPTURE(numerical_integration, p.x1(), dist);
    CHECK(approx_equal(numerical_integration, p.x1(), eps));
  }
  }
  SECTION("backward integration"){
  for (auto const& p : advected_particles) {
    auto const numerical_integration = p.phi()(p.x1(), t0 + tau, -tau);
    auto const dist                  = distance(numerical_integration, p.x0());
    CAPTURE(numerical_integration, p.x0(), dist);
    CHECK(approx_equal(numerical_integration, p.x0(), eps));
  }
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
