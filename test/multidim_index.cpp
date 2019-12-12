#include <tatooine/multidim_index.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
//==============================================================================
TEST_CASE("multidim_index1", "[multidim_index][dynamic]") {
  SECTION("plain_idx") {
    SECTION("x_fastest") {
      dynamic_multidim_index idx{2, 3, 4};
      REQUIRE(idx.plain_idx(0, 0, 0) == 0);
      REQUIRE(idx.plain_idx(1, 0, 0) == 1);
      REQUIRE(idx.plain_idx(0, 1, 0) == 2);
      REQUIRE(idx.plain_idx(0, 0, 1) == 6);
      REQUIRE(idx.plain_idx(0, 1, 1) == 8);
      REQUIRE(idx.plain_idx(1, 1, 1) == 9);
      REQUIRE(idx.plain_idx(1, 2, 3) == idx.num_elements() - 1);

      REQUIRE(idx.in_range(0, 0, 0));
      REQUIRE_FALSE(idx.in_range(-1, 0, 0));
      REQUIRE_FALSE(idx.in_range(0, -1, 0));
      REQUIRE_FALSE(idx.in_range(0, 0, -1));
      REQUIRE_FALSE(idx.in_range(2, 0, 0));
      REQUIRE_FALSE(idx.in_range(0, 3, 0));
      REQUIRE_FALSE(idx.in_range(0, 0, 4));
    }
    SECTION("x_slowest") {
      dynamic_multidim_index<x_slowest> idx{2, 3, 4};
      REQUIRE(idx.plain_idx(0, 0, 0) == 0);
      REQUIRE(idx.plain_idx(1, 0, 0) == 12);
      REQUIRE(idx.plain_idx(0, 1, 0) == 4);
      REQUIRE(idx.plain_idx(0, 0, 1) == 1);
      REQUIRE(idx.plain_idx(0, 1, 1) == 5);
      REQUIRE(idx.plain_idx(1, 1, 1) == 17);
      REQUIRE(idx.plain_idx(1, 2, 3) == idx.num_elements() - 1);

      REQUIRE(idx.in_range(0, 0, 0));
      REQUIRE_FALSE(idx.in_range(-1, 0, 0));
      REQUIRE_FALSE(idx.in_range(0, -1, 0));
      REQUIRE_FALSE(idx.in_range(0, 0, -1));
      REQUIRE_FALSE(idx.in_range(2, 0, 0));
      REQUIRE_FALSE(idx.in_range(0, 3, 0));
      REQUIRE_FALSE(idx.in_range(0, 0, 4));
    }
  }
  SECTION("resize") {
    dynamic_multidim_index idx{2, 3, 4};
    REQUIRE(idx.num_dimensions() == 3);
    idx.resize(1, 2);
    REQUIRE(idx.num_dimensions() == 2);
  }
}
//==============================================================================
TEST_CASE("multidim_index2", "[multidim_index][static]") {
}

//==============================================================================
}  // namespace tatooine
//==============================================================================
