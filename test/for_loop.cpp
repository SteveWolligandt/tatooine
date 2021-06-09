#include <tatooine/for_loop.h>

#include <atomic>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
/// Creates a parallel nested for loop. The outer loop is executed in parallel
TEST_CASE("for_loop_parallel_count", "[for_loop][parallel][count]") {
  SECTION("from zero") {
    std::atomic_size_t cnt       = 0;
    auto               iteration = [&](auto const... /*is*/) { ++cnt; };
    for_loop(iteration, tag::parallel, 100, 100, 100, 100);
    REQUIRE(cnt == 100000000);
  }
  SECTION("from 10 with c-arrays") {
    std::atomic_size_t cnt       = 0;
    auto               iteration = [&](auto const... /*is*/) { ++cnt; };
    for_loop(iteration, tag::parallel, {10, 20}, {10, 20}, {10, 20});
    REQUIRE(cnt == 1000);
  }
}
//------------------------------------------------------------------------------
/// Creates a sequential nested for loop.
TEST_CASE("for_loop_seq", "[for_loop][sequential][count]") {
  SECTION("from zero") {
    size_t i = 0;
    // indices used as parameter pack
    auto iteration = [&](auto... /*is*/) { ++i; };
    for_loop(iteration, tag::sequential, 10, 10, 10);
    REQUIRE(i == 1000);
  }
  SECTION("from 10 with c-arrays") {
    size_t i = 0;
    // indices used as parameter pack
    auto iteration = [&](auto... /*is*/) { ++i; };
    for_loop(iteration, tag::sequential, {10, 20}, {10, 20}, {10, 20});
    REQUIRE(i == 1000);
  }
  SECTION("from 10 with std::pair") {
    size_t i = 0;
    // indices used as parameter pack
    auto iteration = [&](auto... /*is*/) { ++i; };
    for_loop(iteration, tag::sequential, std::pair{10, 20}, std::pair{10, 20},
             std::pair{10, 20});
    REQUIRE(i == 1000);
  }
}
//------------------------------------------------------------------------------
/// Creates a sequential nested for loop and breaks as soon as second index
/// becomes 2.
TEST_CASE("for_loop_break", "[for_loop][sequential][break]") {
  size_t i         = 0;
  auto   iteration = [&](auto const /*i0*/, auto const i1) {
    if (i1 == 2) {
      return false;
    }
    ++i;
    return true;
  };
  for_loop(iteration, tag::sequential, 10, 10);
  REQUIRE(i == 20);
}
//------------------------------------------------------------------------------
TEST_CASE("for_loop_non_zero_begins",
          "[for_loop][sequential][non-zero-begins]") {
  std::vector<std::array<size_t, 2>> collected_indices;
  auto                               iteration = [&](auto const... is) {
    collected_indices.push_back(std::array{is...});
  };
  for_loop(iteration, tag::sequential, {2, 4}, {4, 6});
  REQUIRE(size(collected_indices) == 4);
  REQUIRE(collected_indices[0][0] == 2);
  REQUIRE(collected_indices[0][1] == 4);
  REQUIRE(collected_indices[1][0] == 3);
  REQUIRE(collected_indices[1][1] == 4);
  REQUIRE(collected_indices[2][0] == 2);
  REQUIRE(collected_indices[2][1] == 5);
  REQUIRE(collected_indices[3][0] == 3);
  REQUIRE(collected_indices[3][1] == 5);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
