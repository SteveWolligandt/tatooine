#include <catch2/catch.hpp>
#include <tatooine/parallel_multidim_for.h>

TEST_CASE("parallel_multidim_for") {
  tatooine::parallel_for<3>(1024, 1024, 1024, [](auto is) {});
}
