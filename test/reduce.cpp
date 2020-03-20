#include <tatooine/gpu/reduce.h>
#include <tatooine/random.h>

#include <boost/range/algorithm/generate.hpp>
#include <boost/range/numeric.hpp>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("reduce0", "[gpu][reduce][texture]") {
  std::mt19937 rand_eng{std::random_device{}()};
  random_uniform<float, std::mt19937> rand{rand_eng};
  const size_t                   width = 1024, height = 1024;

  std::vector<float> rand_data(width * height);
  boost::generate(rand_data, [&] { return rand(); });
  yavin::tex2r32f data{rand_data, width, height};

  auto reduced = reduce(data);
  REQUIRE(reduced == boost::accumulate(rand_data, float{0}));
}
//==============================================================================
}
//==============================================================================

