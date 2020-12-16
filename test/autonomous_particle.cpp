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
template <typename V>
auto advect(V const& v, vec2 const& x0, real_t const t0, real_t const t1,
            real_t const radius) {
  real_t const tau_step = 0.01;

  real_t const eps = 1e-3;

  autonomous_particle particle{v, x0, t0, radius};

  auto advected_particles = particle.advect_with_3_splits(tau_step, t1);

  SECTION("forward integration") {
    for (auto& p : advected_particles) {
      auto const numerical_integration = p.phi()(p.x0(), t0, p.t1() - t0);
      auto const dist = distance(numerical_integration, p.x1());
      CAPTURE(numerical_integration, p.x0(), p.x1(), p.t1(), dist, t1);
      CHECK(approx_equal(numerical_integration, p.x1(), eps));
    }
  }
  SECTION("backward integration") {
    for (auto& p : advected_particles) {
      auto const numerical_integration =
          p.phi()(p.x1(), t0 + p.t1(), t0 - p.t1());
      auto const dist = distance(numerical_integration, p.x0());
      CAPTURE(numerical_integration, p.x0(), p.x1(), p.t1(), dist, t1);
      CHECK(approx_equal(numerical_integration, p.x0(), eps));
    }
  }
}
//==============================================================================
template <typename V>
auto advect_until_split(V const& v, vec2 const& x0, real_t const t0,
                        real_t const radius) {
  real_t const max_t    = 1000;
  real_t const tau_step = 0.01;

  real_t const eps = 1e-3;

  autonomous_particle particle{v, x0, t0, radius};

  auto advected_particles = particle.advect_until_split(
      tau_step, max_t, 4,
      std::array{vec2{real_t(1), real_t(1) / real_t(2)},
                 vec2{real_t(1) / real_t(2), real_t(1) / real_t(4)},
                 vec2{real_t(1) / real_t(2), real_t(1) / real_t(4)}},
      std::array{vec2{0, 0}, vec2{0, real_t(3) / 4}, vec2{0, -real_t(3) / 4}});
  REQUIRE(size(advected_particles) > 1);

  SECTION("forward integration") {
    for (auto& p : advected_particles) {
      auto const numerical_integration = p.phi()(p.x0(), t0, p.t1() - t0);
      auto const dist = distance(numerical_integration, p.x1());
      CAPTURE(numerical_integration, p.x0(), p.x1(), p.t1(), dist);
      CHECK(approx_equal(numerical_integration, p.x1(), eps));
    }
  }
  SECTION("backward integration") {
    for (auto& p : advected_particles) {
      auto const numerical_integration =
          p.phi()(p.x1(), t0 + p.t1(), t0 - p.t1());
      auto const dist = distance(numerical_integration, p.x0());
      CAPTURE(numerical_integration, p.x0(), p.x1(), p.t1(), dist);
      CHECK(approx_equal(numerical_integration, p.x0(), eps));
    }
  }
}
//==============================================================================
TEST_CASE("autonomous_particle_doublegyre_first_split",
          "[autonomous_particle][dg][2d][2D][doublegyre]") {
  doublegyre v;
  v.set_infinite_domain(true);
  real_t const t0 = 0;
  real_t const t1 = 5;

  uniform_grid<real_t, 2> g{linspace{0.0, 2.0, 100 + 1},
                            linspace{0.0, 1.0, 50 + 1}};
  g.dimension<0>().pop_front();
  g.dimension<1>().pop_front();
  auto const spacing_x = g.dimension<0>().spacing();
  auto const spacing_y = g.dimension<1>().spacing();
  g.dimension<0>().front() -= spacing_x / 2;
  g.dimension<0>().back() -= spacing_x / 2;
  g.dimension<1>().front() -= spacing_y / 2;
  g.dimension<1>().back() -= spacing_y / 2;
  //real_t const radius = g.dimension<0>().spacing() / 2;
  real_t const radius = 0.01;
  REQUIRE(g.dimension<0>().spacing() == Approx(g.dimension<1>().spacing()));
  for (auto const x0 : g.vertices()) {
    advect_until_split(v, x0, t0, radius);
  }
  for (auto const x0 : g.vertices()) {
    advect(v, x0, t0, t1, radius);
  }
}
//==============================================================================
TEST_CASE("autonomous_particle_saddle_first_split",
          "[autonomous_particle][2d][2D][saddle]") {
  saddle       v;
  vec2         x0{0, 0};
  real_t const t0 = 0;

  real_t const radius = 0.001;
  advect_until_split(v, x0, t0, radius);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
