#include <tatooine/rectilinear_grid.h>
//==============================================================================
auto main() -> int {
  using namespace tatooine;

  auto  rand = random::uniform{};
  auto  grid = rectilinear_grid{linspace{0.0, 1.0, 11}, linspace{0.0, 1.0, 11}};
  auto& prop = grid.scalar_vertex_property("prop");
  grid.vertices().iterate_indices(
      [&](auto const... is) { prop(is...) = rand(); });
  auto sampler = prop.inverse_distance_weighting_sampler(0.1);
  auto resample_grid = rectilinear_grid{linspace{0.0, 1.0, 1001}, linspace{0.0, 1.0, 1001}};
  discretize(sampler, resample_grid, "resampled", execution_policy::parallel);
  grid.write("rectilinear_grid_inverse_distance_weighting_sampler_base.vtr");
  resample_grid.write("rectilinear_grid_inverse_distance_weighting_sampler.vtr");
}
