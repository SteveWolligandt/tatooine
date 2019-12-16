#include <tatooine/parallel_multidim_for.h>

#include <atomic>
#include <catch2/catch.hpp>
#include <thread>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("parallel_multidim_for1") {
  std::atomic_size_t cnt{0};
  auto               f = [&cnt](size_t i) {
    std::cerr << i << '\n';
    ++cnt;
  };
  parallel_for(f, 100);
  REQUIRE(cnt == 100);
}
//==============================================================================
TEST_CASE("parallel_multidim_for2") {
  std::atomic_size_t cnt{0};
  auto               f = [&cnt](auto ix, auto iy) {
    std::cerr << ix << ", " << iy << '\n';
    ++cnt;
  };
  parallel_for(f, 10, 10);
  REQUIRE(cnt == 10 * 10);
}
//==============================================================================
//TEST_CASE("parallel_multidim_for3") {
//  std::atomic_size_t cnt{0};
//  auto               f = [&cnt](auto ix, iy, iz) {
//    std::cerr << ix << ", " << iy << ", " << iz << '\n';
//    ++cnt;
//  };
//  parallel_for(f, 3, 3, 3);
//  REQUIRE(cnt == 3 * 3 * 3);
//}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
