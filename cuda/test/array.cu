#include <tatooine/cuda/array.cuh>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
namespace gpu {
namespace test {
//==============================================================================

TEST_CASE("cuda_array_upload_download_vector",
          "[cuda][array][upload][download][vector]") {
  const std::vector<float> host_data{1, 2, 3, 4};
  cuda::array<float, 1, 2> array{host_data, 2, 2};
  auto downloaded = array.download();
  for (size_t i = 0; i < host_data.size(); ++i) {
    REQUIRE(downloaded[i] == host_data[i]);
  }
}

//==============================================================================
}  // namespace test
}  // namespace gpu
}  // namespace tatooine
//==============================================================================
