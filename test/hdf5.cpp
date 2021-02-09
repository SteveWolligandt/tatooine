#if TATOOINE_HAS_HDF5_SUPPORT
#include <catch2/catch.hpp>
#include <tatooine/grid.h>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("hdf5_grid", "[hdf5][grid]") {
  uniform_grid<double, 3> grid3{2, 2, 2};
  REQUIRE_THROWS(grid3.add_lazy_property<double>("../SDS.h5", "Array"));
  uniform_grid<double, 2> grid2{2, 2};
  REQUIRE_THROWS(grid2.add_lazy_property<double>("../SDS.h5", "Array"));
  uniform_grid<double, 2> grid{100, 100};
  auto &data = grid.add_lazy_property<double>("../SDS.h5", "Array");
  REQUIRE(data(0, 0) == 0);
  REQUIRE(data(99,99) == 1);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
#endif
