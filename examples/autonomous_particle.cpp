#include <tatooine/autonomous_particle.h>
#include <tatooine/analytical/fields/doublegyre.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto v              = analytical::fields::numerical::doublegyre{};
  auto uuid_generator = std::atomic_uint64_t{};
  auto particle = autonomous_particle{vec2{0.5, 0.5}, 0.0, 0.1, uuid_generator};
  auto const [advected_autonomous_particles, advected_simple_particles, mesh] =
      particle.advect_with_three_splits(flowmap(v), 0.01, 0.1, uuid_generator);
  auto discretized_advected_particles =
      std::vector<line2>(size(advected_autonomous_particles));
  std::ranges::copy(
      advected_autonomous_particles | std::views::transform([](auto const& part) {
        return discretize(part, 100);
      }),
      begin(discretized_advected_particles));
  write(discretized_advected_particles, "example_autonomous_particle.vtp");
}
