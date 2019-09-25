#include <tatooine/algorithm.h>
#include <catch2/catch.hpp>
#include <deque>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("algorithm_resize_prev_list", "[algorithm][resize_prev_list]") {
  std::deque d{1,2,3,4};
  auto       it = next(begin(d));
  resize_prev_list(d, it, 3);

  REQUIRE(d.size() == 6);
  REQUIRE(d[2] == 1);
  REQUIRE(d[3] == 2);
  REQUIRE(d[4] == 3);
  REQUIRE(d[5] == 4);

  resize_prev_list(d, it, 0);
  REQUIRE(d.size() == 3);
  REQUIRE(d[0] == 2);
  REQUIRE(d[1] == 3);
  REQUIRE(d[2] == 4);
}

TEST_CASE("algorithm_resize_next_list", "[algorithm][resize_next_list]") {
  std::deque d{1,2,3,4};
  auto       it = next(begin(d));
  resize_next_list(d, it, 3);

  REQUIRE(d.size() == 5);
  REQUIRE(d[0] == 1);
  REQUIRE(d[1] == 2);
  REQUIRE(d[2] == 3);
  REQUIRE(d[3] == 4);

  resize_next_list(d, it, 0);
  REQUIRE(d.size() == 2);
  REQUIRE(d[0] == 1);
  REQUIRE(d[1] == 2);
}


//==============================================================================
}  // namespace tatooine::test
//==============================================================================
