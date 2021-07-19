#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
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
TEST_CASE("autonomous_particle_discretiation_doublegyre",
          "[autonomous_particle][dg][2d][2D][doublegyre][discretization]") {
  auto const initial_radius = 0.1;
  auto const t0             = 0.0;
  auto const t1             = 2.0;
  auto const initial_x      = 1.0;
  auto const tau_step       = 0.1;
  auto const x0             = vec2{initial_x, initial_radius};
  auto const p              = autonomous_particle_2{x0, t0, initial_radius};
  auto const v              = doublegyre{};
  auto       phi            = flowmap(v);
  auto       d = autonomous_particle_flowmap_discretization<double, 2>{
      phi, t1, tau_step, std::deque{p}};
  auto  sampler_grid = grid{linspace{0.0, 2.0, 601}, linspace{0.0, 1.0, 301}};
  auto& s            = sampler_grid.vec2_vertex_property("phi");
  sampler_grid.vertices().iterate_indices([&](auto const... is) {
    try {
      s(is...) = d(sampler_grid.vertex_at(is...), tag::forward);
    } catch (...) {
      s(is...) = vec2{tag::fill{0.0 / 0.0}};
    }
  });
  sampler_grid.write_vtk("autonomous_particle_sampler_doublegyre.vtk");
}
//==============================================================================
TEST_CASE("autonomous_particle_single_doublegyre",
          "[autonomous_particle][dg][2d][2D][doublegyre][single]") {
  auto v = doublegyre{};
  v.set_infinite_domain(true);

  auto const  initial_radius = 0.1;
  auto const  t0             = 0.0;
  auto const  t1             = 2.0;
  auto const  initial_x      = 1.0;
  auto const  tau_step       = 0.1;
  auto const  x0             = vec2{initial_x, initial_radius};
  auto const  p              = autonomous_particle_2{x0, t0, initial_radius};
  auto        phi            = flowmap(v);
  auto const  advected_particles = p.advect_with_3_splits(phi, tau_step, t1);
  auto const& advected_particle  = advected_particles.front();

  auto advected_discretized = discretize(
      advected_particle.ellipse(),
      100);
  for (auto const v : advected_discretized.vertices()) {
    advected_discretized[v] += advected_particle.x1();
  }
  auto initial_discretized  = discretize(
      advected_particle.initial_ellipse(),
      100);
  for (auto const v : initial_discretized.vertices()) {
    initial_discretized[v] += advected_particle.x0();
  }
  auto p_discretized        = discretize(
      p.initial_ellipse(),
      100);
  for (auto const v : p_discretized.vertices()) {
    p_discretized[v] += p.x0();
  }
  advected_discretized.write_vtk("advected_ellipse.vtk");
  initial_discretized.write_vtk("initial_ellipse.vtk");
  p_discretized.write_vtk("original_ellipse.vtk");

  auto sampler = advected_particle.sampler();

  auto const px0 = sampler.B0() * vec2{0.5, 0.5} + sampler.x0();
  auto const px1 = sampler(px0, tag::forward);
  pointset2  ps;
  ps.insert_vertex(px0);
  ps.insert_vertex(px1);
  ps.write_vtk("points.vtk");
  CAPTURE(px0, sampler.x0());
  REQUIRE(sampler.is_inside0(px0));
  CAPTURE(px1, phi(px0, t0, t1 - t0), distance(px1, phi(px0, t0, t1 - t0)));
  REQUIRE(distance(px1, phi(px0, t0, t1 - t0)) < 1e-4);
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
