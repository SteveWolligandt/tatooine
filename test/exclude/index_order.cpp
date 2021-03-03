#include <tatooine/index_order.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("index_order_x_fastest", "[index_order][x_fastest]") {
  using indexing = x_fastest;
  constexpr std::array res0{2,3};
  REQUIRE(indexing::plain_index(res0, 0, 0) == 0);
  REQUIRE(indexing::plain_index(res0, 1, 0) == 1);
  REQUIRE(indexing::plain_index(res0, 0, 1) == 2);
  REQUIRE(indexing::plain_index(res0, 1, 1) == 3);
  REQUIRE(indexing::plain_index(res0, 1, 2) == 5);

  REQUIRE(indexing::multi_index(res0, 0) == std::array{0,0});
  REQUIRE(indexing::multi_index(res0, 1) == std::array{1,0});
  REQUIRE(indexing::multi_index(res0, 2) == std::array{0,1});
  REQUIRE(indexing::multi_index(res0, 3) == std::array{1,1});
  REQUIRE(indexing::multi_index(res0, 4) == std::array{0,2});
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

  //REQUIRE(indexing::multi_index(res0, 0) == std::array{0, 0});
  //REQUIRE(indexing::multi_index(res0, 1) == std::array{1, 0});
  //REQUIRE(indexing::multi_index(res0, 2) == std::array{0, 1});
  //REQUIRE(indexing::multi_index(res0, 3) == std::array{1, 1});
  //REQUIRE(indexing::multi_index(res0, 4) == std::array{0, 2});
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
