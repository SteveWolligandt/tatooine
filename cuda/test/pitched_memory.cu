#include <tatooine/cuda/pitched_memory.cuh>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================
TEST_CASE("pitched_memory0", "[pitched_memory][download][2d]") {
  pitched_memory<float, 2> buffer(std::vector<float>{1, 2, 3, 4, 5, 6}, 2, 3);
  auto transformed_data = buffer.download();
  REQUIRE(transformed_data[0] == 1);
  REQUIRE(transformed_data[1] == 2);
  REQUIRE(transformed_data[2] == 3);
  REQUIRE(transformed_data[3] == 4);
  REQUIRE(transformed_data[4] == 5);
  REQUIRE(transformed_data[5] == 6);
}
//==============================================================================
__global__ void pitched_memory1(pitched_memory<float, 2> buffer) {
  auto globalIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                              blockIdx.y * blockDim.y + threadIdx.y);
  if (globalIdx.x >= buffer.width() || globalIdx.y >= buffer.height()) { return; }
  buffer(globalIdx) += globalIdx.x + globalIdx.y;
};
TEST_CASE("pitched_memory1", "[pitched_memory][transform][2d]") {
  pitched_memory<float, 2> buffer(std::vector<float>{1, 2, 3, 4, 5, 6}, 2, 3);
  pitched_memory1<<<dim3(2,3), dim3(1,1)>>>(buffer);
  auto transformed_data = buffer.download();
  REQUIRE(transformed_data[0] == 1 + 0 + 0);
  REQUIRE(transformed_data[1] == 2 + 1 + 0);
  REQUIRE(transformed_data[2] == 3 + 0 + 1);
  REQUIRE(transformed_data[3] == 4 + 1 + 1);
  REQUIRE(transformed_data[4] == 5 + 0 + 2);
  REQUIRE(transformed_data[5] == 6 + 1 + 2);
}
//==============================================================================
TEST_CASE("pitched_memory2", "[pitched_memory][3d][download]") {
  pitched_memory<float, 3> buffer(
      std::vector<float>{1, 2, 3, 4, 5, 6, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f},
      2, 3, 2);
  auto transformed_data = buffer.download();
  REQUIRE(transformed_data[0]  == 1.0f);
  REQUIRE(transformed_data[1]  == 2.0f);
  REQUIRE(transformed_data[2]  == 3.0f);
  REQUIRE(transformed_data[3]  == 4.0f);
  REQUIRE(transformed_data[4]  == 5.0f);
  REQUIRE(transformed_data[5]  == 6.0f);
  REQUIRE(transformed_data[6]  == 1.5f);
  REQUIRE(transformed_data[7]  == 2.5f);
  REQUIRE(transformed_data[8]  == 3.5f);
  REQUIRE(transformed_data[9]  == 4.5f);
  REQUIRE(transformed_data[10] == 5.5f);
  REQUIRE(transformed_data[11] == 6.5f);
}
//==============================================================================
}  // namespace test
}  // namespace cuda
}  // namespace tatooine
//==============================================================================
