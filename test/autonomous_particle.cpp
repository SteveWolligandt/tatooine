#include <tatooine/autonomous_particle.h>
#include <tatooine/center_field.h>
#include <tatooine/concepts.h>
#include <tatooine/doublegyre.h>
#include <tatooine/saddle_field.h>
#include <tatooine/vtk_legacy.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
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

      autonomous_particle p0{x0, t0, radius};
      auto const          particles = p0.integrate(v, tau_step, t1);

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
  double const max_distance = 1e-10;

  autonomous_particle const p0{x0, t0, radius};
  auto const                particles = p0.integrate(v, tau_step, t1);

  auto integrator = p0.create_integrator();
  size_t i = 0;
  for (auto const& particle : particles) {
    auto const total_integration_length = t0 - particle.t1();
    if (total_integration_length < 0) {
      if constexpr (has_analytical_flowmap_v<V>) {
        auto flowmap = v->flowmap();
        parameterized_line<VReal, 3, interpolation::linear> integral_curve;
        for (auto const tau : linspace<VReal>{total_integration_length, VReal{0}, 10}) {
          auto const x =
              flowmap(particle.x1(), particle.t1(), tau);
          integral_curve.push_back(vec{x(0), x(1), particle.t1()+tau}, particle.t1()+tau);
        }
        integral_curve.write_vtk(name + "_backintegration_" +
                                 std::to_string(i++) + ".vtk");
      } else {
        integration::vclibs::rungekutta43<VReal, 3, interpolation::linear>
                        st_integrator;
        spacetime_field st_v{v};
        st_integrator
            .integrate_uncached(
                st_v, vec{particle.x1(0), particle.x1(1), particle.t1()},
                particle.t1(), total_integration_length, 0)
            .write_vtk(name + "_backintegration_" + std::to_string(i++) +
                       ".vtk");
      }
    }

    auto const back_integration =
        integrator.integrate_uncached(v, particle.x1(), particle.t1(), total_integration_length)
            .front_vertex();

    auto const distance = tatooine::distance(back_integration, particle.x0());
    CAPTURE(particle.x1(), particle.t1(), total_integration_length, particle.level(), particle.x0(),
            back_integration, distance);
    CHECK(distance < max_distance);
  }
}
//==============================================================================
TEST_CASE("autonomous_particle_dg_vtk",
          "[autonomous_particle][dg][doublegyre][vtk]") {
  grid                  g{linspace{0.0, 2.0, 11}, linspace{0.0, 1.0, 6}};
  numerical::doublegyre v;
  v.set_infinite_domain(true);
  autonomous_particle_write_vtk("dg", v, g, 0, 5, 0.1);
}
//------------------------------------------------------------------------------
TEST_CASE("autonomous_particle_dg_backward_integration",
          "[autonomous_particle][dg][doublegyre][backward_integration]") {
  grid const            g{linspace{0.0, 2.0, 11}, linspace{0.0, 1.0, 6}};
  numerical::doublegyre v;
  v.set_infinite_domain(true);

  autonomous_particles_test_backward_integation_distance("dg", v, g(2, 4), g.spacing(0),
                                                         0, 5, 0.1);
}
//==============================================================================
TEST_CASE("autonomous_particle_saddle_vtk",
          "[autonomous_particle][saddle][vtk]") {
  grid g{linspace{-1.0, 1.0, 11}, linspace{-1.0, 1.0, 11}};
  numerical::saddle_field v;
  autonomous_particle_write_vtk("saddle", v, g, 0, 2, 0.1);
}
//------------------------------------------------------------------------------
TEST_CASE("autonomous_particle_saddle_backward_integration",
          "[autonomous_particle][saddle][backward_integration]") {
  grid g{linspace{-1.0, 1.0, 11}, linspace{-1.0, 1.0, 11}};
  numerical::saddle_field v;
  autonomous_particles_test_backward_integation_distance("saddle", v, g(5, 5),
                                                         g.spacing(0), 0, 2, 0.1);
}
//==============================================================================
TEST_CASE("autonomous_particle_center_vtk",
          "[autonomous_particle][center][vtk]") {
  grid                  g{linspace{-1.0, 1.0, 11}, linspace{-1.0, 1.0, 11}};
  numerical::center_field v;
  autonomous_particle_write_vtk("center", v, g, 0, 5, 0.1);
}
//------------------------------------------------------------------------------
TEST_CASE("autonomous_particle_center_backward_integration",
          "[autonomous_particle][center][backward_integration]") {
  grid g{linspace{-1.0, 1.0, 11}, linspace{-1.0, 1.0, 11}};
  numerical::center_field v;
  autonomous_particles_test_backward_integation_distance("center", v, vec{1.0, 1.0}, g.spacing(0),
                                                         0, 5, 0.1);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
