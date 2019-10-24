#include <catch2/catch.hpp>
#include <tatooine/cuda/buffer.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================

TEST_CASE("cuda_global_buffer_upload_download_vector",
          "[cuda][buffer][upload][download][vector]") {
  const std::vector<float> v1{1.0f, 2.0f};
  buffer<float>      dv1(v1);
  const auto               cv1 = dv1.download();
  for (size_t i = 0; i < v1.size(); ++i) { REQUIRE(v1[i] == cv1[i]); }
}
//==============================================================================
TEST_CASE("cuda_global_buffer_upload_download_initializer_list",
          "[cuda][buffer][upload][download][initializer_list]") {
  buffer<float> dv1{1.0f, 2.0f};
  const auto          cv1 = dv1.download();
  REQUIRE(1.0f == cv1[0]);
  REQUIRE(2.0f == cv1[1]);
}
//==============================================================================
__global__ void kernel(buffer<float> in, buffer<float> out) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < in.size()) { out[i] = in[i]; }
}
TEST_CASE("cuda_global_buffer_kernel_copy", "[cuda][buffer][kernel]") {
  buffer<float> in{1, 2, 3, 4, 5};
  buffer<float> out(5);
  kernel<<<5, 1>>>(in, out);
  auto hout = out.download();
  for (size_t i = 0; i < 5; ++i) {
    INFO(i);
    REQUIRE(hout[i] == i + 1);
  }
}

//==============================================================================
}  // namespace test
}  // namespace gpu
}  // namespace tatooine
//==============================================================================
