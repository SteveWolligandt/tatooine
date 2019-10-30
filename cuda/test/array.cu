#include <catch2/catch.hpp>
#include <tatooine/cuda/array.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================

TEST_CASE("cuda_array0", "[cuda][array][upload][download][vector]") {
  const std::vector<float> host_data{1, 2, 3, 4};
  cuda::array<float, 1, 2> array{host_data, 2, 2};
  auto                     res = array.resolution();
  REQUIRE(res[0] == 2);
  REQUIRE(res[1] == 2);
  const auto downloaded = array.download();
  REQUIRE(downloaded.size() == host_data.size());
  for (size_t i = 0; i < host_data.size(); ++i) {
    REQUIRE(host_data[i] == downloaded[i]);
  }
  array
}

TEST_CASE("cuda_array1", "[cuda][array][upload][download][vector]") {
  const std::vector<float> host_data{1, 2, 3, 4, 1.5, 2.5, 3.5, 4.5};
  cuda::array<float, 1, 3> array{host_data, 2, 2, 2};
  auto                     res = array.resolution();
  REQUIRE(res[0] == 2);
  REQUIRE(res[1] == 2);
  REQUIRE(res[2] == 2);
  const auto downloaded = array.download();
  REQUIRE(downloaded.size() == host_data.size());
  for (size_t i = 0; i < host_data.size(); ++i) {
    REQUIRE(host_data[i] == downloaded[i]);
  }
  array.free();
}

//==============================================================================
}  // namespace test
}  // namespace cuda
}  // namespace tatooine
//==============================================================================
