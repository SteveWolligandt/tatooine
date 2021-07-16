#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/concepts.h>
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/spacetime_vectorfield.h>
#include <tatooine/vtk_legacy.h>

#include <catch2/catch.hpp>
#include <fstream>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace analytical::fields::numerical;
//==============================================================================
TEST_CASE("autonomous_particle_doublegyre",
          "[autonomous_particle][dg][2d][2D][doublegyre]") {
  auto v = doublegyre{};
  v.set_infinite_domain(true);

  auto const  initial_radius = 0.0001;
  auto const  t0             = 0.0;
  auto const  t1             = 2.0;
  auto const  initial_x      = 1.0;
  auto const  p   = autonomous_particle_2{vec2{initial_x, initial_radius}, t0,
                                       initial_radius};
  auto        phi = flowmap(v);
  auto const  advected_particles = p.advect_with_3_splits(phi, 0.1, t1);
  auto const& advected_particle  = advected_particles.front();

  auto advected_discretized = discretize(advected_particle.ellipse(), 100);
  auto initial_discretized =
      discretize(advected_particle.initial_ellipse(), 100);
  auto p_discretized = discretize(p.initial_ellipse(), 100);
  for (auto const v : advected_discretized.vertices()) {
    advected_discretized[v] += advected_particle.x1();
  }
  for (auto const v : initial_discretized.vertices()) {
    initial_discretized[v] += advected_particle.x0();
  }
  for (auto const v : p_discretized.vertices()) {
    p_discretized[v] += p.x0();
  }
  advected_discretized.write_vtk("advected_ellipse.vtk");
  initial_discretized.write_vtk("initial_ellipse.vtk");
  p_discretized.write_vtk("original_ellipse.vtk");

  auto const ell0 = advected_particle.initial_ellipse();
  auto const ell1 = advected_particle.ellipse();


  auto const B0 = ell0.main_axes();
  auto const B1 = ell1.main_axes();

  auto const x0 = B0 * vec2{0.5, 0.5} + advected_particle.x0();
  REQUIRE(ell0.is_inside(x0 - advected_particle.x0()));

  auto const x1 =
      B1 * *inv(B0) * (x0 - advected_particle.x0()) + advected_particle.x1();
  pointset2 ps;
  ps.insert_vertex(x0);
  ps.insert_vertex(x1);
  ps.write_vtk("points.vtk");
  CAPTURE(x1, phi(x0, t0, t1 - t0), distance(x1, phi(x0, t0, t1 - t0)));
  REQUIRE(distance(x1, phi(x0, t0, t1 - t0)) < 1e-4);
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
