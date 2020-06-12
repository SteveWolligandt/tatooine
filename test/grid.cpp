#include <tatooine/grid.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("grid_vertex_indexing", "[grid][grid_vertex][indexing]") {
  std::array  dim0{0, 1, 2};
  std::vector dim1{0, 1, 2};
  linspace    dim2{0.0, 1.0, 10};
  grid        g{dim0, dim1, dim2};
  //const auto v00 = g.vertex_at(0,0);
  //const auto v11 = g.vertex_at(1,1);
  //const auto v42 = g.vertex_at(4,2);
  //
  //REQUIRE(*v00[0] == 0);
  //REQUIRE(*v00[1] == 0);
  //REQUIRE(approx_equal(v00.position(), vec{0.0, 0.0}));
  //REQUIRE(*v11[0] == 1);
  //REQUIRE(*v11[1] == 1);
  //REQUIRE(approx_equal(v11.position(), vec{1.0, 1.0}));
  //REQUIRE(*v42[0] == 4);
  //REQUIRE(*v42[1] == 2);
  //REQUIRE(approx_equal(v42.position(), vec{4.0, 2.0}));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
