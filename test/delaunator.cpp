#include <tatooine/delaunator.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
TEST_CASE("delaunator", "[delaunator]"){
  std::vector<vec2> points{{-1, 1}, {1, 1}, {1, -1}, {-1, -1}};
  delaunator::Delaunator d{points};

  for (size_t i = 0; i < d.triangles.size(); i += 3) {
    std::cerr << "Triangle points: ["
              << points[d.triangles[i]] << ", "
              << points[d.triangles[i + 1]] << ", "
              << points[d.triangles[i + 2]] << "]\n";
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
