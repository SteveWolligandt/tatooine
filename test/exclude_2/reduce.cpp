#include <tatooine/gpu/reduce.h>
#include <tatooine/random.h>

#include <boost/range/numeric.hpp>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("reduce0", "[gpu][reduce][texture]") {
  using res_container = std::vector<std::pair<int, int>>;
  auto seed           = std::random_device{}();
  std::mt19937 eng{seed};
  for (const auto [width, height] : res_container{{16, 16},
                                                  {32, 32},
                                                  {64, 64},
                                                  {128, 128},
                                                  {256, 256},
                                                  {512, 512},
                                                  {1024, 1024},
                                                  {871, 1290}}) {
    CAPTURE(width, height, seed);
    const std::vector<float> rand_data =
        random::uniform_vector<float>(width * height, 0.0f, 1.0f, eng);
    const rendering::gl::tex2r32f data_tex{rand_data, width, height};
    const auto reduced              = reduce(data_tex, 16, 16);
    REQUIRE(reduced == Approx(boost::accumulate(rand_data, float{0})));
  }
}
//==============================================================================
TEST_CASE("reduce1", "[gpu][reduce][texture]") {
  using res_container = std::vector<size_t>;
  auto seed           = std::random_device{}();
  std::mt19937 eng{seed};
  for (const auto size : res_container{1024, 2048}) {
    CAPTURE(size, seed);
    const std::vector<float> rand_data =
        random::uniform_vector<float>(size, 0.0f, 1.0f, eng);
    const rendering::gl::shaderstoragebuffer<float> in_buffer{rand_data};
    const auto                       reduced = reduce(in_buffer, 16*16);
    REQUIRE(reduced == Approx(boost::accumulate(rand_data, float{0})));
  }
}
//==============================================================================
}
//==============================================================================

