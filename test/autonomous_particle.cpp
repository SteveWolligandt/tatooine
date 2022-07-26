#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
#include <tatooine/concepts.h>
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/spacetime_vectorfield.h>
#include <tatooine/pointset.h>
#include <tatooine/vtk_legacy.h>

#include <catch2/catch.hpp>
#include <fstream>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace analytical::numerical;
//==============================================================================
//struct autonomous_particle_fixture : autonomous_particle2 {
//  static constexpr auto initial_radius = 0.1;
//  static constexpr auto t0             = 0.0;
//  static constexpr auto t1             = 1.0;
//  static constexpr auto initial_x      = 1.0;
//  static constexpr auto tau_step       = 0.5;
//  static constexpr auto initial_pos    = vec2{initial_x, initial_radius};
//  autonomous_particle_fixture()
//      : autonomous_particle2{initial_pos, t0, initial_radius} {}
//};
////==============================================================================
//TEST_CASE_METHOD(
//    autonomous_particle_fixture,
//    "autonomous_particle_discretiation_doublegyre_single",
//    "[autonomous_particle][dg][2d][2D][doublegyre][single][discretization]") {
//  auto const v   = doublegyre{};
//  auto       phi = flowmap(v);
//  auto       d   = autonomous_particle_flowmap_discretization2{
//      phi, t1, tau_step, std::deque<autonomous_particle2>{*this}};
//  auto  sampler_grid = rectilinear_grid{linspace{0.0, 2.0, 2001}, linspace{0.0, 1.0, 1001}};
//  auto& s            = sampler_grid.vec2_vertex_property("phi");
//  sampler_grid.vertices().iterate_indices([&](auto const... is) {
//    try {
//      s(is...) = d(sampler_grid.vertex_at(is...), forward);
//    } catch (...) {
//      s(is...) = vec2{tag::fill{0.0 / 0.0}};
//    }
//  });
//  sampler_grid.write_vtk("autonomous_particle_sampler_doublegyre.vtk");
//}
////==============================================================================
//TEST_CASE("autonomous_particle_discretiation_doublegyre_rectilinear_grid",
//          "[autonomous_particle][dg][2d][2D][doublegyre][discretization]"
//          "[rectilinear]") {
//  auto const v   = doublegyre{};
//  auto       phi = flowmap(v);
//  auto       d   = autonomous_particle_flowmap_discretization2{
//      phi, autonomous_particle_fixture::t0, autonomous_particle_fixture::t1,
//      autonomous_particle_fixture::tau_step,
//      rectilinear_grid{linspace{0.0, 2.0, 101}, linspace{0.0, 1.0, 51}}};
//  auto sampler_grid =
//      rectilinear_grid{linspace{0.0, 2.0, 2001}, linspace{0.0, 1.0, 1001}};
//  auto& phi_forward  = sampler_grid.vec2_vertex_property("phi_forward");
//  auto& phi_backward = sampler_grid.vec2_vertex_property("phi_backward");
//  sampler_grid.vertices().iterate_indices(
//      [&](auto const... is) {
//        auto const x = sampler_grid.vertex_at(is...);
//        try {
//          phi_forward(is...) = d(x, forward);
//        } catch (...) {
//          phi_forward(is...) = vec2{tag::fill{0.0 / 0.0}};
//        }
//        try {
//          phi_backward(is...) = d(x, backward);
//        } catch (...) {
//          phi_backward(is...) = vec2{tag::fill{0.0 / 0.0}};
//        }
//      },
//      execution_policy::sequential);
//  sampler_grid.write_vtk("autonomous_particle_sampler_doublegyre.vtk");
//  std::vector<line2> all_advected_discretizations;
//  std::vector<line2> all_initial_discretizations;
//  for (auto const& sampler : d.samplers()) {
//    all_initial_discretizations.push_back(discretize(sampler.ellipse0(), 100));
//    all_advected_discretizations.push_back(discretize(sampler.ellipse1(), 100));
//  }
//  write_vtk(all_initial_discretizations, "all_initial_ellipses.vtk");
//  write_vtk(all_advected_discretizations, "all_advected_ellipses.vtk");
//}
////==============================================================================
//TEST_CASE_METHOD(autonomous_particle_fixture,
//                 "autonomous_particle_single_doublegyre",
//                 "[autonomous_particle][dg][2d][2D][doublegyre][single]") {
//  auto v = doublegyre{};
//  v.set_infinite_domain(true);
//
//  auto        phi                = flowmap(v);
//  auto const  advected_particles = advect_with_3_splits(phi, tau_step, t1);
//  auto const& advected_particle  = advected_particles.front();
//
//  auto advected_discretized = discretize(advected_particle, 100);
//  auto initial_discretized =
//      discretize(advected_particle.initial_ellipse(), 100);
//  std::vector<line2> all_advected_discretizations;
//  std::vector<line2> all_initial_discretizations;
//  for (auto const& advected_particle : advected_particles) {
//    all_initial_discretizations.push_back(
//        discretize(advected_particle.initial_ellipse(), 100));
//    all_advected_discretizations.push_back(discretize(advected_particle, 100));
//  }
//  auto p_discretized = discretize(initial_ellipse(), 100);
//  advected_discretized.write_vtk("advected_ellipse.vtk");
//  initial_discretized.write_vtk("initial_ellipse.vtk");
//  write_vtk(all_initial_discretizations, "all_initial_ellipses.vtk");
//  write_vtk(all_advected_discretizations, "all_advected_ellipses.vtk");
//  p_discretized.write_vtk("original_ellipse.vtk");
//
//  auto sampler = advected_particle.sampler();
//
//  auto const px0 = sampler.ellipse0().S() * vec2{0.5, 0.5} + sampler.ellipse0().center();
//  auto const px1 = sampler(px0, forward);
//  pointset2  ps;
//  ps.insert_vertex(px0);
//  ps.insert_vertex(px1);
//  ps.write_vtk("points.vtk");
//  CAPTURE(px0, sampler.ellipse0().center());
//  REQUIRE(sampler.is_inside0(px0));
//  CAPTURE(px1, phi(px0, t0, t1 - t0), distance(px1, phi(px0, t0, t1 - t0)));
//  REQUIRE(distance(px1, phi(px0, t0, t1 - t0)) < 1e-4);
//}
//==============================================================================
TEST_CASE("autonomous_particl_identity_flowmap",
          "[autonomous_particle][identity_flowmap]") {
  auto const v              = doublegyre{};
  auto     const  phi            = flowmap(v);
  auto       uuid_generator = std::atomic_uint64_t{};
  auto const t0 = 0;
  auto const r0 = 0.01;
  auto const t_end = 2;
  auto     const  part = autonomous_particle2{vec2{1, 0.5}, t0, r0, uuid_generator};
  auto const [advected_autonomous_particles, advected_single_particles,
              reconstructed_neighbors] =
      part.advect_with_three_splits(phi, 1e-4, t_end, uuid_generator);
  for (auto const& p : advected_autonomous_particles) {
    auto const s = p.sampler();
    SECTION("compare with self") {
    //  SECTION("forward") {
        CAPTURE(s(s.x0(forward), forward), s.x0(backward));
        REQUIRE(approx_equal(s(s.x0(forward), forward), s.x0(backward)));
      //}
      //SECTION("backward") {
        CAPTURE(s(s.x0(backward), backward), s.x0(forward));
        REQUIRE(approx_equal(s(s.x0(backward), backward), s.x0(forward)));
    //  }
    }

    SECTION("compare with numerical integration") {
      auto const eps = 5e-4;
      SECTION("forward") {
        CAPTURE(s.x0(backward), phi(s.x0(forward), t0, t_end - t0));
        REQUIRE(approx_equal(s.x0(backward), phi(s.x0(forward), t0, t_end - t0),
                             eps));
      }
      SECTION("backward") {
        CAPTURE(s.x0(forward), phi(s.x0(backward), t_end, t0 - t_end));
        REQUIRE(approx_equal(s.x0(forward),
                             phi(s.x0(backward), t_end, t0 - t_end), eps));
      }
    }
  }
}
//==============================================================================
TEST_CASE("autonomous_particle_post_triangulation_simple_cases",
          "[autonomous_particle][post_triangulation][simple_cases]") {
  using namespace detail::autonomous_particle;
  SECTION("(∙ ∙)") {
    auto edges = edgeset2{};
    auto v0    = edges.insert_vertex(0, 0);
    auto v1    = edges.insert_vertex(1, 0);
    auto map   = std::unordered_map<std::uint64_t, edgeset2::vertex_handle>{};
    map[1]     = v0;
    map[2]     = v1;
    auto hierarchy_pairs = std::vector<hierarchy_pair>{{0, 0}, {1, 0}, {2, 0}};
    auto h               = hierarchy{hierarchy_pairs, map, edges};
    triangulate(edges, h);

    CAPTURE(edges.edges().size());
    REQUIRE(edges.are_connected(v0, v1));
    REQUIRE(edges.edges().size() == 1);
  }
  SECTION("∙ ⋮") {
    auto edges = edgeset2{};
    auto v0    = edges.insert_vertex(0, 0);
    auto v1    = edges.insert_vertex(1, 1);
    auto v2    = edges.insert_vertex(1, 0);
    auto v3    = edges.insert_vertex(1, -1);
    auto map   = std::unordered_map<std::uint64_t, edgeset2::vertex_handle>{};
    map[1]     = v0;
    map[3]     = v1;
    map[4]     = v2;
    map[5]     = v3;
    auto hierarchy_pairs = std::vector<hierarchy_pair>{{0, 0}, {1, 0}, {2, 0},
                                                       {3, 2}, {4, 2}, {5, 2}};
    auto h               = hierarchy{hierarchy_pairs, map, edges};
    triangulate(edges, h);

    CAPTURE(edges.edges().size());
    REQUIRE(edges.are_connected(v0, v1));
    REQUIRE(edges.are_connected(v0, v2));
    REQUIRE(edges.are_connected(v0, v3));
    REQUIRE(edges.are_connected(v1, v2));
    REQUIRE(edges.are_connected(v2, v3));
    REQUIRE(edges.edges().size() == 5);
  }
  SECTION("∙ ∙∙∙") {
    auto edges = edgeset2{};
    auto v0    = edges.insert_vertex(0, 0);
    auto v1    = edges.insert_vertex(1, 0);
    auto v2    = edges.insert_vertex(2, 0);
    auto v3    = edges.insert_vertex(3, 0);
    auto map   = std::unordered_map<std::uint64_t, edgeset2::vertex_handle>{};
    map[1]     = v0;
    map[3]     = v1;
    map[4]     = v2;
    map[5]     = v3;
    auto hierarchy_pairs = std::vector<hierarchy_pair>{{0, 0}, {1, 0}, {2, 0},
                                                       {3, 2}, {4, 2}, {5, 2}};
    auto h               = hierarchy{hierarchy_pairs, map, edges};
    triangulate(edges, h);

    CAPTURE(edges.edges().size());
    REQUIRE(edges.are_connected(v0, v1));
    REQUIRE(edges.are_connected(v1, v2));
    REQUIRE(edges.are_connected(v2, v3));
    REQUIRE(edges.edges().size() == 3);
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
