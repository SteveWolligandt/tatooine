#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
//==============================================================================
using namespace tatooine;
//=============================================================================
auto advect(auto const& v, auto& particles, auto& uuid_generator, auto const t0,
            auto const t_end) {
  auto       phi = flowmap(v);

  auto i = std::size_t{};
  auto t_ends = linspace{t0, t_end, static_cast<std::size_t>((t_end - t0) * 10)};
  t_ends.pop_front();
  for (auto const t : t_ends) {
    write_vtp(particles, 33, "animation_" + std::to_string(i) + ".vtp",
              backward);
    particles = std::get<0>(autonomous_particle2::advect_with_three_splits(
        phi, 0.01, t, particles, uuid_generator));
    ++i;
  }
  write_vtp(particles, 33, "animation_" + std::to_string(i) + ".vtp", backward);
}
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto                        uuid_generator = std::atomic_uint64_t{};
  [[maybe_unused]] auto const r              = 0.01;
  [[maybe_unused]] auto const t0             = real_number(0);
  [[maybe_unused]] auto       t_end          = real_number(4);
  if (argc > 1) {
    t_end = std::stod(argv[1]);
  }
  //============================================================================
  auto dg = analytical::fields::numerical::doublegyre{};
  dg.set_infinite_domain(true);

  auto const eps                  = 1e-3;
  auto initial_particles_dg = autonomous_particle2::particles_from_grid(
      t0,
      rectilinear_grid{linspace{0.0 + eps, 2.0 - eps, 61},
                       linspace{0.0 + eps, 1.0 - eps, 31}},
      uuid_generator);
  advect(dg, initial_particles_dg, uuid_generator, t0, t_end);
}
