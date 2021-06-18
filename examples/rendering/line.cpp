#include <tatooine/rendering/line.h>
using namespace tatooine;
auto main() -> int {
  auto  ls   = std::vector{line3{vec3{0, 0, 0}, vec3{1, 1, 0}, vec3{2, 0, 0}}};
  auto &prop = ls.front().scalar_vertex_property("prop");
  prop[line3::vertex_handle{0}] = 1;
  prop[line3::vertex_handle{1}] = 2;
  prop[line3::vertex_handle{2}] = 30;
  ls.front().compute_parameterization();

  ls.push_back(ls.front().resample<interpolation::cubic>(linspace{
      ls.front().parameterization()[ls.front().vertices().front()],
      ls.front().parameterization()[ls.front().vertices().back()], 101}));
  rendering::interactive(ls);
}
