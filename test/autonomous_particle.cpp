#include <tatooine/analytical/fields/numerical/autonomous_particles_test.h>
#include <tatooine/analytical/fields/numerical/center.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/concepts.h>
#include <tatooine/spacetime_vectorfield.h>
#include <tatooine/vtk_legacy.h>

#include <catch2/catch.hpp>
#include <fstream>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace analytical::fields::numerical;
//==============================================================================
template <typename V>
auto advect_until_time(std::ofstream& forward_logger,
                       std::ofstream& backward_logger, V const& v,
                       vec2 const& x0, real_t const t0, real_t const t1,
                       real_t const radius,
                       real_t const max_num_particles = 0) {
  real_t const tau_step = 0.5;
  real_t const eps      = 1e-3;

  auto phi = flowmap(v);

  autonomous_particle particle{x0, t0, radius};
  STATIC_REQUIRE(
      std::is_same_v<decltype(particle), autonomous_particle<double, 2>>);
  STATIC_REQUIRE(
      std::is_same_v<autonomous_particle<double, 2>,
                     typename decltype(particle)::container_t::value_type>);

  bool stop = false;
  auto advected_particles =
      particle.advect_with_3_splits(phi, tau_step, t1, max_num_particles, stop);

  for (auto& p : advected_particles) {
    auto const numerical_integration = phi(p.x0(), t0, p.t1() - t0);
    auto const dist                  = distance(numerical_integration, p.x1());
    CAPTURE(size(advected_particles), numerical_integration, p.x0(), p.x1(),
            p.t1(), dist, t1);
    CHECK(std::abs(dist) < eps);
    forward_logger << t1 - t0 << "," << radius << "," << dist << '\n';
  }
  for (auto& p : advected_particles) {
    auto const numerical_integration = phi(p.x1(), t0 + p.t1(), t0 - p.t1());
    auto const dist                  = distance(numerical_integration, p.x0());
    CAPTURE(size(advected_particles), numerical_integration, p.x0(), p.x1(),
            p.t1(), dist, t1);
    CHECK(std::abs(dist)< eps);
    backward_logger << t1 - t0 << "," << radius << "," << dist << '\n';
  }
}
//==============================================================================
template <typename V>
auto advect_until_split(V const& v, vec2 const& x0, real_t const t0,
                        real_t const radius) {
  real_t const max_t    = 1000;
  real_t const tau_step = 0.5;
  real_t const eps      = 1e-4;

  auto                phi = flowmap(v);
  autonomous_particle particle{x0, t0, radius};

  auto advected_particles_first_step = particle.advect_until_split(
      phi, tau_step, max_t, 4,
      std::array{vec2{real_t(1), real_t(1) / real_t(2)},
                 vec2{real_t(1) / real_t(2), real_t(1) / real_t(4)},
                 vec2{real_t(1) / real_t(2), real_t(1) / real_t(4)}},
      std::array{vec2{0, 0}, vec2{0, real_t(3) / 4}, vec2{0, -real_t(3) / 4}});

  {
    for (auto& p : advected_particles_first_step) {
      auto const numerical_integration = phi(p.x0(), t0, p.t1() - t0);
      auto const dist = distance(numerical_integration, p.x1());
      CAPTURE(numerical_integration, p.x0(), p.x1(), p.t1(), dist, eps);
      CHECK(std::abs(dist) < eps);
    }
  }
  {
    for (auto& p : advected_particles_first_step) {
      auto const numerical_integration = phi(p.x1(), t0 + p.t1(), t0 - p.t1());
      auto const dist = distance(numerical_integration, p.x0());
      CAPTURE(numerical_integration, p.x0(), p.x1(), p.t1(), dist, eps);
      CHECK(std::abs(dist) < eps);
    }
  }

  for (auto pf : advected_particles_first_step) {
    auto advected_particles_second_step = pf.advect_until_split(
        phi, tau_step, max_t, 4,
        std::array{vec2{real_t(1), real_t(1) / real_t(2)},
                   vec2{real_t(1) / real_t(2), real_t(1) / real_t(4)},
                   vec2{real_t(1) / real_t(2), real_t(1) / real_t(4)}},
        std::array{vec2{0, 0}, vec2{0, real_t(3) / 4},
                   vec2{0, -real_t(3) / 4}});

    {
      for (auto& p : advected_particles_second_step) {
        auto const numerical_integration = phi(p.x0(), t0, p.t1() - t0);
        auto const dist = distance(numerical_integration, p.x1());
        CAPTURE(numerical_integration, p.x0(), p.x1(), p.t1(), dist), eps;
        CHECK(std::abs(dist) < eps);
      }
    }
    {
      for (auto& p : advected_particles_second_step) {
        auto const numerical_integration =
            phi(p.x1(), t0 + p.t1(), t0 - p.t1());
        auto const dist = distance(numerical_integration, p.x0());
        CAPTURE(numerical_integration, p.x0(), p.x1(), p.t1(), dist, eps);
        CHECK(std::abs(dist) < eps);
      }
    }
  }
}
//------------------------------------------------------------------------------
auto error_estimation(std::ofstream& forward_logger,
                      std::ofstream& backward_logger, auto const& v,
                      auto const& g) {
  std::vector<real_t> radii{1e-4};
  std::vector<real_t> t0s{0};
  std::vector<real_t> t1s{20};
  size_t const max     = size(radii) * size(t0s) * size(t1s) * g.num_vertices();
  size_t       cnt     = 0;
  auto         monitor = std::thread{[&] {
    while (cnt < max) {
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
      std::cerr << (double)cnt / max * 100 << "%               \r";
    }
    std::cerr << '\n';
  }};
  for (auto const radius : radii) {
    for (auto const t0 : t0s) {
      for (auto const t1 : t1s) {
        for (auto const x0 : g.vertices()) {
          advect_until_time(forward_logger, backward_logger, v, x0, t0, t1,
                            radius, 100);
          ++cnt;
        }
      }
    }
  }
  monitor.join();
}
//==============================================================================
TEST_CASE("autonomous_particle_doublegyre",
          "[autonomous_particle][dg][2d][2D][doublegyre]") {
  std::ofstream forward_logger{"autonomous_particles_doublegyre_forward.csv",
                               std::ios::app},
      backward_logger{"autonomous_particles_doublegyre_backward.csv",
                      std::ios::app};
  forward_logger << "tau,radius,error\n";
  backward_logger << "tau,radius,error\n";
  doublegyre v;
  v.set_infinite_domain(true);

  uniform_grid<real_t, 2> g{linspace{0.0, 2.0, 10 + 1},
                            linspace{0.0, 1.0, 5 + 1}};
  auto const              spacing_x = g.dimension<0>().spacing();
  auto const              spacing_y = g.dimension<1>().spacing();
  g.dimension<0>().front() -= spacing_x / 2;
  g.dimension<0>().back() -= spacing_x / 2;
  g.dimension<1>().front() -= spacing_y / 2;
  g.dimension<1>().back() -= spacing_y / 2;
  g.dimension<0>().pop_back();
  g.dimension<1>().pop_back();
  g.dimension<0>().pop_back();
  g.dimension<1>().pop_back();

  error_estimation(forward_logger, backward_logger, v, g);
}
//==============================================================================
// TEST_CASE("autonomous_particle_saddle",
//          "[autonomous_particle][2d][2D][saddle][until_split]") {
//  saddle                  v;
//  uniform_grid<real_t, 2> g{linspace{-1.0, 1.0, 10}, linspace{-1.0, 1.0,
//  10}}; error_estimation(v, g);
//}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
