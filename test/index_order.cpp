#include <tatooine/index_order.h>
#include <catch2/catch_test_macros.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("index_order_x_fastest", "[index_order][x_fastest]") {
  using indexing = x_fastest;
  SECTION("2D") {
    constexpr std::array res{2, 3};
    REQUIRE(indexing::plain_index(res, 0, 0) == 0);
    REQUIRE(indexing::plain_index(res, 1, 0) == 1);
    REQUIRE(indexing::plain_index(res, 0, 1) == 2);
    REQUIRE(indexing::plain_index(res, 1, 1) == 3);
    REQUIRE(indexing::plain_index(res, 1, 2) == 5);

    REQUIRE(indexing::multi_index(res, 0)[0] == 0);
    REQUIRE(indexing::multi_index(res, 0)[0] == 0);
    REQUIRE(indexing::multi_index(res, 1)[0] == 1);
    REQUIRE(indexing::multi_index(res, 1)[1] == 0);
    REQUIRE(indexing::multi_index(res, 2)[0] == 0);
    REQUIRE(indexing::multi_index(res, 2)[1] == 1);
    REQUIRE(indexing::multi_index(res, 3)[0] == 1);
    REQUIRE(indexing::multi_index(res, 3)[1] == 1);
    REQUIRE(indexing::multi_index(res, 4)[0] == 0);
    REQUIRE(indexing::multi_index(res, 4)[1] == 2);
    REQUIRE(indexing::multi_index(res, 5)[0] == 1);
    REQUIRE(indexing::multi_index(res, 5)[1] == 2);
  }
  SECTION("3D") {
    SECTION("2,3,4") {
      constexpr std::array res{2, 3, 4};
      REQUIRE(indexing::plain_index(res, 0, 0, 0) == 0);
      REQUIRE(indexing::plain_index(res, 1, 0, 0) == 1);
      REQUIRE(indexing::plain_index(res, 0, 1, 0) == 2);
      REQUIRE(indexing::plain_index(res, 0, 0, 1) == 6);
      {
        auto const is = indexing::multi_index(res, 0);
        REQUIRE(is[0] == 0);
        REQUIRE(is[1] == 0);
        REQUIRE(is[2] == 0);
      }
      {
        auto const is = indexing::multi_index(res, 1);
        REQUIRE(is[0] == 1);
        REQUIRE(is[1] == 0);
        REQUIRE(is[2] == 0);
      }
      {
        auto const is = indexing::multi_index(res, 2);
        REQUIRE(is[0] == 0);
        REQUIRE(is[1] == 1);
        REQUIRE(is[2] == 0);
      }
      {
        auto const is = indexing::multi_index(res, 6);
        REQUIRE(is[0] == 0);
        REQUIRE(is[1] == 0);
        REQUIRE(is[2] == 1);
      }
    }
    SECTION("5,8,1") {
      constexpr std::array res{5, 8, 1};
      {
        auto const is = indexing::multi_index(res, 32);
        REQUIRE(is[0] == 2);
        REQUIRE(is[1] == 6);
        REQUIRE(is[2] == 0);
      }
    }
  }
}
//==============================================================================
TEST_CASE("index_order_x_slowest", "[index_order][x_slowest]") {
  using indexing = x_slowest;
  constexpr std::array res0{2,3};
  REQUIRE(indexing::plain_index(res0, 0, 0) == 0);
  REQUIRE(indexing::plain_index(res0, 1, 0) == 3);
  REQUIRE(indexing::plain_index(res0, 0, 1) == 1);
  REQUIRE(indexing::plain_index(res0, 1, 1) == 4);
  REQUIRE(indexing::plain_index(res0, 1, 2) == 5);

  //REQUIRE(indexing::multi_index(res0, 0)[0] == 0);
  //REQUIRE(indexing::multi_index(res0, 0)[0] == 0);
  //REQUIRE(indexing::multi_index(res0, 1)[0] == 0);
  //REQUIRE(indexing::multi_index(res0, 1)[1] == 1);
  //REQUIRE(indexing::multi_index(res0, 2)[0] == 0);
  //REQUIRE(indexing::multi_index(res0, 2)[1] == 2);
  //REQUIRE(indexing::multi_index(res0, 3)[0] == 1);
  //REQUIRE(indexing::multi_index(res0, 3)[1] == 0);
  //REQUIRE(indexing::multi_index(res0, 4)[0] == 1);
  //REQUIRE(indexing::multi_index(res0, 4)[1] == 1);
  //REQUIRE(indexing::multi_index(res0, 5)[0] == 1);
  //REQUIRE(indexing::multi_index(res0, 5)[1] == 2);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
