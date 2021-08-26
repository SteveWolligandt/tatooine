#include <tatooine/pointset.h>
//==============================================================================
using namespace tatooine;
static auto constexpr num_points = 100;
//==============================================================================
auto main() -> int {
  using boost::generate;
  auto ps   = pointset2{};
  auto rand = random::uniform{-1.0, 1.0};
  for (size_t i = 0; i < num_points; ++i) {
    ps.insert_vertex(vec2{rand});
  }
  auto& scalars = ps.scalar_vertex_property("scalars");
  generate(scalars, [&rand] { return rand(); });
  ps.write("example_moving_least_squares_sampler_data.vtk");
  auto const x_range        = linspace{-1.0, 1.0, 100};
  auto const y_range        = x_range;
  auto       sampler_domain = rectilinear_grid{x_range, y_range};
  auto       sampler        = ps.moving_least_squares_sampler(scalars, 0.01);
  discretize(sampler, sampler_domain, "scalars", 0, tag::sequential);
  sampler_domain.write("example_moving_least_squares_sampler.vtk");
}
