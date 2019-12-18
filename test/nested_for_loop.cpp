#include <tatooine/nested_for_loop.h>
#include <catch2/catch.hpp>
#include <atomic>

//==============================================================================
namespace tatooine::test {
//==============================================================================
/// Creates a parallel nested for loop. The outer loop is executed in parallel
TEST_CASE("nested_for_loop_par"){
  std::atomic_size_t cnt       = 0;
  // indices used as single parameters
  auto iteration = [&](auto /*i0*/, auto /*i1*/, auto /*i2*/, auto /*i3*/) { ++cnt; };
  parallel_nested_for(iteration, 100, 100, 100, 100);
  REQUIRE(cnt == 100000000);
}
//------------------------------------------------------------------------------
/// Creates a sequential nested for loop.
TEST_CASE("nested_for_loop_seq"){
  size_t i = 0;
  // indices used as parameter pack
  auto iteration = [&](auto... /*is*/) { ++i; };
  nested_for(iteration, 10, 10, 10);
  REQUIRE(i == 1000);
}
//------------------------------------------------------------------------------
/// Creates a sequential nested for loop and breaks as soon as second index
/// becomes 2.
TEST_CASE("nested_for_loop_break"){
  size_t i = 0;
  auto iteration = [&](auto /*i0*/, auto i1) {
    if (i1 == 2) { return false; }
    ++i;
    return true;
  };
  nested_for(iteration, 10, 10);
  REQUIRE(i == 20);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
