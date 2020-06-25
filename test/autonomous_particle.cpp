#include <tatooine/analytical/fields/numerical/autonomous_particles_test.h>
#include <tatooine/analytical/fields/numerical/center.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/concepts.h>
#include <tatooine/vtk_legacy.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace analytical::fields::numerical;
template <typename V, std::floating_point VReal, std::floating_point GridReal>
void autonomous_particle_write_vtk(std::string const&              name,
                                   vectorfield<V, VReal, 2> const& v,
                                   grid<GridReal, 2> const&        g,
                                   arithmetic auto t0, arithmetic auto t1,
                                   arithmetic auto tau_step) {
  REQUIRE(g.spacing(0) == Approx(g.spacing(1)));
  double const radius = g.spacing(0);
  for (size_t y = 1; y < g.size(1) - 1; ++y) {
    for (size_t x = 1; x < g.size(0) - 1; ++x) {
      auto const x0 = g(x, y);

      autonomous_particle p0{v, x0, t0, radius};
      auto const          particles = p0.integrate(tau_step, t1);

      write_vtk(particles, t0,
                name + "_autonomous_particle_paths_forward" +
                    std::to_string(x) + "_" + std::to_string(y) + ".vtk",
                name + "_autonomous_particle_paths_backward" +
                    std::to_string(x) + "_" + std::to_string(y) + ".vtk");
    }
  }
}
//------------------------------------------------------------------------------
template <typename V, std::floating_point VReal, std::floating_point X0Real>
void autonomous_particles_test_backward_integation_distance(
    std::string const& name, vectorfield<V, VReal, 2> const& v,
    vec<X0Real, 2> const& x0, std::floating_point auto const radius,
    arithmetic auto t0, arithmetic auto t1, arithmetic auto tau_step) {
  double const eps = 1e-10;
  std::vector<parameterized_line<VReal, 3, interpolation::linear>>
                            integral_curves;
  autonomous_particle const p0{v, x0, t0, radius};
  for (auto const& particle : p0.integrate(tau_step, t1)) {
    auto const     total_integration_length = t0 - particle.t1();
    auto&          integral_curve           = integral_curves.emplace_back();
    decltype(auto) phi                      = particle.get_flowmap();
    auto           stepped_back_integration0 =
        phi(particle.x1(), particle.t1(), total_integration_length);
    for (auto const tau : linspace<VReal>{
             total_integration_length, VReal{0},
             static_cast<size_t>(50 * std::abs(total_integration_length))}) {
      auto const x = phi(particle.x1(), particle.t1(), tau);
      integral_curve.push_back(vec{x(0), x(1), particle.t1() + tau},
                               particle.t1() + tau);
    }

    auto const back_integration = particle.get_flowmap()(
        particle.x1(), particle.t1(), total_integration_length);

    {
      INFO("Runge Kutta back step in initial circle")
      CAPTURE(particle.x1(), particle.t1(), total_integration_length,
              particle.level(), particle.x0(), particle.start_radius(),
              stepped_back_integration0);
      CHECK(tatooine::distance(stepped_back_integration0, p0.x0()) <=
            particle.start_radius());
    }

    //{
    //  INFO("Formula back step in initial circle")
    //  CAPTURE(particle.x1(), particle.t1(), total_integration_length,
    //          particle.level(), particle.x0(), particle.start_radius());
    //  CHECK(tatooine::distance(back_integration, p0.x0()) < p0.start_radius());
    //}
    //
    //{
    //  INFO("Runge Kutta back step and formula back step approximately equal")
    //  CAPTURE(particle.x1(), particle.t1(), total_integration_length,
    //          particle.level(), particle.x0(),particle.start_radius(), back_integration);
    //  CHECK(tatooine::distance(back_integration, particle.x0()) < eps);
    //}
  }

  write_vtk(integral_curves, name + "_backintegration_.vtk");
}
//------------------------------------------------------------------------------
template <typename V, std::floating_point VReal, std::floating_point X0Real>
void autonomous_particles_test_backward_integation_distance(
    std::string const& name, vectorfield<V, VReal, 2> const& v,
    std::vector<vec<X0Real, 2>> const& x0s,
    std::floating_point auto const radius, arithmetic auto t0,
    arithmetic auto t1, arithmetic auto tau_step) {
  double const eps = 1e-10;
  std::vector<parameterized_line<VReal, 3, interpolation::linear>>
      integral_curves;
  for (auto const& x0 : x0s) {
    autonomous_particle const p0{v, x0, t0, radius};
    for (auto const& particle : p0.integrate(tau_step, t1)) {
      auto const total_integration_length = t0 - particle.t1();
      if (total_integration_length <= t0 - t1 + 1e-4) {
        auto& integral_curve = integral_curves.emplace_back();
        for (auto const tau :
             linspace<VReal>{total_integration_length, VReal{0}, 10}) {
          auto const x =
              particle.get_flowmap()(particle.x1(), particle.t1(), tau);
          integral_curve.push_back(vec{x(0), x(1), particle.t1() + tau},
                                   particle.t1() + tau);
        }
      }

      auto const back_integration = particle.get_flowmap()(
          particle.x1(), particle.t1(), total_integration_length);

      auto const distance = tatooine::distance(back_integration, particle.x0());
      CAPTURE(particle.x1(), particle.t1(), total_integration_length,
              particle.level(), particle.x0(), back_integration, distance);
      CHECK(distance < eps);
    }
  }

  write_vtk(integral_curves, name + "_backintegration_.vtk");
}
//==============================================================================
TEST_CASE("autonomous_particle_dg_vtk",
          "[autonomous_particle][dg][doublegyre][vtk]") {
  grid const g{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}};
  doublegyre v;
  v.set_infinite_domain(true);
  autonomous_particle_write_vtk("dg", v, g, 0, 10, 0.1);
}
//------------------------------------------------------------------------------
TEST_CASE("autonomous_particle_dg_backward_integration",
          "[autonomous_particle][dg][doublegyre][backward_integration]") {
  // grid const g{linspace{0.0, 2.0, 21}, linspace{0.0, 1.0, 11}};
  doublegyre v;
  v.set_infinite_domain(true);

  // std::vector<vec<double, 2>> x0s;
  // for (size_t y = 1; y < g.size(1) - 1; ++y) {
  //
  //  for (size_t x = 1; x < g.size(0) - 1; ++x) { x0s.push_back(g(x, y)); }
  //}
  // autonomous_particles_test_backward_integation_distance(
  //    "dg", v, x0s, g.spacing(0), 0, 10, 0.1);
  autonomous_particles_test_backward_integation_distance("dg", v, vec{1.0, 0.5},
                                                         0.001, 0, 10, 0.1);
}
//==============================================================================
TEST_CASE("autonomous_particle_saddle_vtk",
          "[autonomous_particle][saddle][vtk]") {
  grid const g{linspace{-1.0, 1.0, 11}, linspace{-1.0, 1.0, 11}};
  autonomous_particle_write_vtk("saddle", saddle{}, g, 0, 2, 0.1);
}
//------------------------------------------------------------------------------
TEST_CASE("autonomous_particle_saddle_backward_integration",
          "[autonomous_particle][saddle][backward_integration]") {
  grid const g{linspace{-1.0, 1.0, 11}, linspace{-1.0, 1.0, 11}};
  autonomous_particles_test_backward_integation_distance(
      "saddle", saddle{}, std::vector{g(5, 5)}, g.spacing(0), 0, 2, 0.1);
}
//==============================================================================
TEST_CASE("autonomous_particle_center_vtk",
          "[autonomous_particle][center][vtk]") {
  grid const g{linspace{-1.0, 1.0, 11}, linspace{-1.0, 1.0, 11}};
  autonomous_particle_write_vtk("center", center{}, g, 0, 5, 0.1);
}
//------------------------------------------------------------------------------
TEST_CASE("autonomous_particle_center_backward_integration",
          "[autonomous_particle][center][backward_integration]") {
  grid const g{linspace{-1.0, 1.0, 11}, linspace{-1.0, 1.0, 11}};
  autonomous_particles_test_backward_integation_distance(
      "center", center{}, std::vector{g(5, 5)}, g.spacing(0), 0, 5, 0.1);
}
//==============================================================================
TEST_CASE("autonomous_particle_test_field_vtk",
          "[autonomous_particle][test_field][vtk]") {
  grid const g{linspace{-1.0, 1.0, 3}, linspace{-1.0, 1.0, 3}};
  autonomous_particle_write_vtk("test_field", autonomous_particles_test{}, g, 0,
                                5, 0.1);
}
//------------------------------------------------------------------------------
TEST_CASE("autonomous_particle_test_field_backward_integration",
          "[autonomous_particle][test_field][backward_integration]") {
  grid const                  g{linspace{-1.0, 1.0, 3}, linspace{-1.0, 1.0, 3}};
  std::vector<vec<double, 2>> x0s;
  for (size_t y = 1; y < g.size(1) - 1; ++y) {
    for (size_t x = 1; x < g.size(0) - 1; ++x) { x0s.push_back(g(x, y)); }
  }
  autonomous_particles_test_backward_integation_distance(
      "test_field", autonomous_particles_test{}, x0s, g.spacing(0), 0, 3, 0.1);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
