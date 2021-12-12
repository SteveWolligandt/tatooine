#include <tatooine/geometry/ellipsoid.h>
using namespace tatooine;
auto main() -> int {
  //auto e = geometry::ellipsoid{vec3{1, 1, 1}, vec3{-1, -1, -1}, vec3{-1, 1, 1}};
  auto e    = geometry::ellipsoid{1.0, 1.0, 1.0};
  auto disc = discretize(e, 2);
  disc.remove(decltype(disc)::vertex_handle{0});
  disc.write_vtp("ellipsoid.example.vtp");
}
