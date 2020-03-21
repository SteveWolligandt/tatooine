#include <tatooine/gpu/reduce.h>
#include <tatooine/random.h>

#include <boost/range/numeric.hpp>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("reduce0", "[gpu][reduce][texture]") {
  constexpr size_t      width = 1024, height = 1024;
  auto            rand_data = random_uniform_vector<float>(width * height);
  yavin::tex2r32f data{rand_data, width, height};

  auto reduced = reduce(data);
  REQUIRE(reduced == boost::accumulate(rand_data, float{0}));
}
//==============================================================================
}
//==============================================================================

