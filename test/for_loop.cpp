#include <tatooine/for_loop.h>

#include <atomic>
#include <catch2/catch_test_macros.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
#if TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
/// Creates a parallel nested for loop. The outer loop is executed in parallel
TEST_CASE("for_loop_parallel_count", "[for_loop][parallel][count]") {
  SECTION("from zero") {
    auto cnt       = std::atomic_size_t{};
    auto iteration = [&](auto const... /*is*/) { ++cnt; };
    for_loop(iteration, execution_policy::parallel, 10, 10, 10, 10);
    REQUIRE(cnt == 10000);
  }
  SECTION("from 10 with c-arrays") {
    std::atomic_size_t cnt       = 0;
    auto               iteration = [&](auto const... /*is*/) { ++cnt; };
    for_loop(iteration, execution_policy::parallel, {10, 20}, {10, 20},
             {10, 20});
    REQUIRE(cnt == 1000);
  }
}
#endif
//------------------------------------------------------------------------------
/// Creates a sequential nested for loop.
TEST_CASE("for_loop_seq", "[for_loop][sequential][count]") {
  SECTION("from zero") {
    auto i = std::size_t {};
    // indices used as parameter pack
    auto iteration = [&](auto const... /*is*/) { ++i; };
    for_loop(iteration, execution_policy::sequential,
             10,
             10,
             10);
    REQUIRE(i == 1000);
  }
  SECTION("from 10 with c-arrays") {
    auto i = std::size_t{};
    // indices used as parameter pack
    auto iteration = [&](auto const... /*is*/) { ++i; };
    for_loop(iteration, execution_policy::sequential,
             {10, 20},
             {10, 20},
             {10, 20});
    REQUIRE(i == 1000);
  }
  SECTION("from 10 with std::pair") {
    auto i = std::size_t{};
    // indices used as parameter pack
    auto iteration = [&](auto const... /*is*/) { ++i; };
    for_loop(iteration, execution_policy::sequential,
             std::pair{10, 20},
             std::pair{10, 20},
             std::pair{10, 20});
    REQUIRE(i == 1000);
  }
}
//------------------------------------------------------------------------------
/// Creates a sequential nested for loop and breaks as soon as second index
/// becomes 2.
TEST_CASE("for_loop_break", "[for_loop][sequential][break]") {
  std::size_t i         = 0;
  auto        iteration = [&](auto const /*i0*/, auto const i1) {
    if (i1 == 2) {
      return false;
    }
    ++i;
    return true;
  };
  for_loop(iteration, execution_policy::sequential, 10, 10);
  REQUIRE(i == 20);
}
//------------------------------------------------------------------------------
TEST_CASE("for_loop_non_zero_begins",
          "[for_loop][sequential][non-zero-begins]") {
  std::vector<std::array<std::size_t, 2>> collected_indices;
  auto                                    iteration = [&](auto const... is) {
    collected_indices.push_back(std::array{is...});
  };
  for_loop(iteration, execution_policy::sequential, {2, 4}, {4, 6});
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
//------------------------------------------------------------------------------
TEST_CASE("for_loop_pair_list", "[for_loop][sequential][pair_list]") {
  auto collected_indices = std::vector<std::vector<std::size_t>>{};
  auto iteration         = [&](std::vector<std::size_t> const& is) {
    collected_indices.push_back(is);
  };
  auto const ranges = std::vector{std::pair{std::size_t(2), std::size_t(4)},
                                  std::pair{std::size_t(4), std::size_t(6)}};
  for_loop(iteration, ranges);
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
TEST_CASE("for_loop_generic_range_sequential",
          "[for_loop][sequential][generic_range]") {
  auto const range = std::vector{1.0, 2.0, 3.0, 4.0};
  auto acc = double{};
  auto       iteration = [&](auto const& elem) { acc += elem; };
  for_loop(iteration, range, execution_policy::sequential);
  REQUIRE(acc == 10.0);
}
//==============================================================================
#if TATOOINE_PARALLEL_FOR_LOOPS_AVAILABLE
TEST_CASE("for_loop_generic_range_parallel",
          "[for_loop][parallel][generic_range]") {
  auto const range = std::vector{1.0, 2.0, 3.0, 4.0};
  auto acc = std::atomic<double>{};
  auto       iteration = [&](auto const& elem) { acc += elem; };
  for_loop(iteration, range, execution_policy::parallel);
  REQUIRE(acc == 10.0);
}
#endif
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
