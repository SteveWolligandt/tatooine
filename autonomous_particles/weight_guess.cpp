#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
//==============================================================================
using namespace tatooine;
auto constexpr advection_direction = forward;
//==============================================================================
using autonomous_particle_flowmap_type =
    AutonomousParticleFlowmapDiscretization<
        2, AutonomousParticle<2>::split_behaviors::five_splits>;
//==============================================================================
auto doit(auto& g, auto const& v, auto const& initial_particles,
          auto& uuid_generator, auto const t0, auto const t_end,
          auto const inverse_distance_num_samples){};
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto                        uuid_generator = std::atomic_uint64_t{};
  [[maybe_unused]] auto const r              = 0.01;
  [[maybe_unused]] auto const t0             = double(0);
  [[maybe_unused]] auto       t_end          = double(1);
  [[maybe_unused]] auto       inverse_distance_num_samples = std::size_t(5);
  if (argc > 1) {
    t_end = std::stod(argv[1]);
  }
  if (argc > 2) {
    inverse_distance_num_samples = std::stoi(argv[2]);
  }
  //============================================================================
  auto       dg                   = analytical::fields::numerical::doublegyre{};
  auto const eps                  = 1e-3;
  auto const initial_particles= autonomous_particle2::particles_from_grid(
      t0,
      rectilinear_grid{linspace{0.0 + eps, 2.0 - eps, 61},
                       linspace{0.0 + eps, 1.0 - eps, 31}},
      uuid_generator);

  auto const tau = t_end - t0;
  auto       phi = flowmap(dg);

  auto flowmap_autonomous_particles = autonomous_particle_flowmap_type{
      phi, t_end, 0.01, initial_particles, uuid_generator};

  auto  ps  = pointset2{};
  auto& phi_prop = ps.vec2_vertex_property("phi");

  auto const q     = vec2{1.0, 0.1};
  auto const q_phi = phi(q, t0, tau);

  for (auto const& s : flowmap_autonomous_particles.samplers()) {
    auto v = ps.insert_vertex(s.x0(advection_direction));
    phi_prop[v] = s(q, advection_direction);
  }


  auto [nearest_vertices, squared_distances] = ps.nearest_neighbors_radius(q, 0.1);

  std::cout << "q=[" << q(0) << ", " << q(1) << "];\n";
  std::cout << "q_phi=[" << q_phi(0) << ", " << q_phi(1) << "];\n";

  std::cout << "d=[";
  std::cout << squared_distances[0];
  for (std::size_t i = 1; i < squared_distances.size(); ++i) {
    std::cout << ";"<< squared_distances[i] ;
  }
  std::cout << "];\n";

      std::cout
            << "Phi=[";
  for (auto const v : nearest_vertices) {
    std::cout << phi_prop[v](0) << "," << phi_prop[v](1) << ";";
  }
  std::cout << "];\n";
}
