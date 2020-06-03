#include <tatooine/autonomous_particle.h>
#include <tatooine/doublegyre.h>
#include <tatooine/vtk_legacy.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("autonomnous_particle0", "[autonomous_particle]") {
  grid g{linspace{0.0, 2.0, 10}, linspace{0.0, 1.0, 5}};
  numerical::doublegyre v;
  v.set_infinite_domain(true);
  double const t0       = 0;
  double const t1       = 5;
  double const tau_step = 0.1;
  double const radius   = 2.0 / 10;
  for (size_t y = 1; y < g.size(1) - 1; ++y) {
    for (size_t x = 1; x < g.size(0) - 1; ++x) {
      auto const x0 = g(x, y);

      autonomous_particle p0{x0, t0, radius};
      auto const          particles = p0.integrate(v, tau_step, t1);

      write_vtk(particles, t0,
                "autonomous_particle_paths_forward" + std::to_string(x) + "_" +
                    std::to_string(y) + ".vtk",
                "autonomous_particle_paths_backward" + std::to_string(x) + "_" +
                    std::to_string(y) + ".vtk");
    }
  }
}
//==============================================================================
TEST_CASE("autonomnous_particle1", "[autonomous_particle][backward_integration]") {
  grid g{linspace{0.0, 2.0, 10}, linspace{0.0, 1.0, 5}};
  numerical::doublegyre v;
  v.set_infinite_domain(true);
  double const max_distance = 2e-1;
  double const t0           = 0;
  double const t1           = 5;
  double const tau_step     = 0.1;
  double const radius       = 2.0 / 10;
  vec const    x0{1.0, 0.3};

  autonomous_particle const p0{x0, t0, radius};
  auto const                particles = p0.integrate(v, tau_step, t1);

  auto integrator = p0.create_integrator();
  for (auto const& particle:particles) {
    auto const tau = t0-particle.t1();
    auto const back_integration =
        integrator.integrate_uncached(v, particle.x1(), particle.t1(), tau)
            .front_vertex();
    auto const distance = tatooine::distance(back_integration, particle.x1());
    CAPTURE(particle.x1(), particle.t1(), tau, particle.level(), particle.x0(),
            back_integration, distance);
    CHECK(distance < max_distance);
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
