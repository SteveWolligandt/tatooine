#include <tatooine/lazy_netcdf_reader.h>
#include <tatooine/grid.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("lazy_netcdf_reader", "[lazy_netcdf_reader]") {
  netcdf::lazy_reader<double> cont{"simple_xy.nc", "data", {2, 2}};
  auto const&                 resolution = cont.size();
  CAPTURE(resolution);
  SECTION("accessing first chunk") {
    SECTION("accessing first element") {
      REQUIRE(cont.chunk_at_is_null(0, 0));
      REQUIRE(cont(0, 0) == 0);
      REQUIRE_FALSE(cont.chunk_at_is_null(0, 0));
    }
    SECTION("accessing second element") {
      REQUIRE(cont.chunk_at_is_null(0, 0));
      REQUIRE(cont(1, 0) == 1);
      REQUIRE_FALSE(cont.chunk_at_is_null(0, 0));
    }
  }
  SECTION("accessing last chunk") {
    REQUIRE(cont.chunk_at_is_null(3, 2));
    REQUIRE(cont(resolution[0] - 1, resolution[1] - 1) == 47);
    REQUIRE_FALSE(cont.chunk_at_is_null(3, 2));
  }
  SECTION("correct order") {
    REQUIRE(cont(resolution[0] - 1, 0) == 7);
    REQUIRE(cont(0, resolution[1] - 1) == 40);
  }
}
//==============================================================================
TEST_CASE("lazy_netcdf_reader_grid", "[lazy_netcdf_reader][grid]") {
  non_uniform_grid<double, 2> g{"simple_xy.nc"};
  auto const& prop = g.vertex_property<double>("data");

  REQUIRE(prop(7, 0) == Approx(7));
  REQUIRE(prop(0, 5) == Approx(40));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
