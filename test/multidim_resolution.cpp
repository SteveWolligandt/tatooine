#include <tatooine/multidim_resolution.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
//==============================================================================
TEST_CASE("dynamic_multidim_resolution", "[multidim_resolution][dynamic]") {
  SECTION("plain_idx") {
    SECTION("x_fastest") {
      dynamic_multidim_resolution idx{2, 3, 4};
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
      dynamic_multidim_resolution<x_slowest> idx{2, 3, 4};
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
    dynamic_multidim_resolution idx{2, 3, 4};
    REQUIRE(idx.num_dimensions() == 3);
    idx.resize(1, 2);
    REQUIRE(idx.num_dimensions() == 2);
  }
}
//==============================================================================
TEST_CASE("static_multidim_resolution", "[multidim_resolution][static]") {
  SECTION("plain_idx") {
    SECTION("x_fastest") {
      using idx = static_multidim_resolution<x_fastest, 2,3,4>;
      REQUIRE(idx::plain_idx(0, 0, 0) == 0);
      REQUIRE(idx::plain_idx(1, 0, 0) == 1);
      REQUIRE(idx::plain_idx(0, 1, 0) == 2);
      REQUIRE(idx::plain_idx(0, 0, 1) == 6);
      REQUIRE(idx::plain_idx(0, 1, 1) == 8);
      REQUIRE(idx::plain_idx(1, 1, 1) == 9);
      REQUIRE(idx::plain_idx(1, 2, 3) == idx::num_elements() - 1);

      REQUIRE(idx::in_range(0, 0, 0));
      REQUIRE_FALSE(idx::in_range(-1, 0, 0));
      REQUIRE_FALSE(idx::in_range(0, -1, 0));
      REQUIRE_FALSE(idx::in_range(0, 0, -1));
      REQUIRE_FALSE(idx::in_range(2, 0, 0));
      REQUIRE_FALSE(idx::in_range(0, 3, 0));
      REQUIRE_FALSE(idx::in_range(0, 0, 4));
    }
    SECTION("x_slowest") {
      using idx = static_multidim_resolution<x_slowest, 2,3,4>;
      REQUIRE(idx::plain_idx(0, 0, 0) == 0);
      REQUIRE(idx::plain_idx(1, 0, 0) == 12);
      REQUIRE(idx::plain_idx(0, 1, 0) == 4);
      REQUIRE(idx::plain_idx(0, 0, 1) == 1);
      REQUIRE(idx::plain_idx(0, 1, 1) == 5);
      REQUIRE(idx::plain_idx(1, 1, 1) == 17);
      REQUIRE(idx::plain_idx(1, 2, 3) == idx::num_elements() - 1);

      REQUIRE(idx::in_range(0, 0, 0));
      REQUIRE_FALSE(idx::in_range(-1, 0, 0));
      REQUIRE_FALSE(idx::in_range(0, -1, 0));
      REQUIRE_FALSE(idx::in_range(0, 0, -1));
      REQUIRE_FALSE(idx::in_range(2, 0, 0));
      REQUIRE_FALSE(idx::in_range(0, 3, 0));
      REQUIRE_FALSE(idx::in_range(0, 0, 4));
    }
  }
}

//==============================================================================
}  // namespace tatooine
//==============================================================================
