#include <tatooine/cuda/global_buffer.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
namespace gpu {
namespace test {
//==============================================================================

TEST_CASE("cuda_global_buffer_upload_download_vector",
          "[cuda][global_buffer][upload][download][vector]") {
  const std::vector<float> v1{1.0f, 2.0f};
  cuda::global_buffer<float> dv1(v1);
  const auto cv1 = dv1.download();
  for (size_t i = 0; i < v1.size(); ++i) { REQUIRE(v1[i] == cv1[i]); }
}
//==============================================================================
TEST_CASE("cuda_global_buffer_upload_download_initializer_list",
          "[cuda][global_buffer][upload][download][initializer_list]") {
  cuda::global_buffer<float> dv1{1.0f, 2.0f};
  const auto cv1 = dv1.download();
  REQUIRE(1.0f == cv1[0]);
  REQUIRE(2.0f == cv1[1]);
}
//==============================================================================
__global__ void kernel(float* in, float *out, size_t size) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) { out[i] = in[i]; }
}
TEST_CASE("cuda_global_buffer_kernel_copy",
          "[cuda][global_buffer][kernel]") {
  cuda::global_buffer<float> dv_in{1,2,3,4,5};
  cuda::global_buffer<float> dv_out(5);
  kernel<<<5,1>>>(dv_in.device_ptr(), dv_out.device_ptr(), 5);
  auto hv_out = dv_out.download();
  for (size_t i = 0; i < 5; ++i) { INFO(i); REQUIRE(hv_out[i] == i+1); }
}

//==============================================================================
}  // namespace tatooine
}  // namespace gpu
}  // namespace test
//==============================================================================
