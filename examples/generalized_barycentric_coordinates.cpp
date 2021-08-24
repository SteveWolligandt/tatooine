#include <tatooine/polygon.h>
#include <string>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto const x            = std::stod(argv[1]);
  auto const y            = std::stod(argv[2]);
  auto const q            = vec2{x, y};
  auto       polygon_data = std::vector<vec2>{};
  for (int i = 3; i < argc; i += 2) {
    polygon_data.emplace_back(std::stod(argv[i]), std::stod(argv[i + 1]));
  }

  auto       p                       = polygon2{std::move(polygon_data)};
  for (auto const b : p.barycentric_coordinates(q)) {
    std::cout << b << '\n';
  }
}
