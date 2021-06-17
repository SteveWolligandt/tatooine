#include <tatooine/rendering/line.h>
using namespace tatooine;
auto main() -> int {
  auto  l         = line3{vec3{0, 0, 0}, vec3{1, 1, 0}, vec3{2, 0, 0}};
  auto &prop      = l.scalar_vertex_property("prop");
  using handle    = line2::vertex_handle;
  prop[line3::vertex_handle{0}] = 1;
  prop[line3::vertex_handle{1}] = 2;
  prop[line3::vertex_handle{2}] = 30;
  l.compute_parameterization();

  auto const resampled = l.resample<interpolation::cubic>(
      linspace{l.parameterization()[l.vertices().front()],
               l.parameterization()[l.vertices().back()], 101});
  rendering::interactive(resampled);
}
