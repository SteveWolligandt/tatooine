#include <iostream>
#include <tatooine/cuda/texture_buffer.h>
#include <tatooine/cuda/global_buffer.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
namespace gpu {
namespace test {
//==============================================================================

__global__ void test_kernel0(float* out, cudaTextureObject_t tex, int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const float u = float(i) / (width-1);
  const float v = float(j) / (height-1);
  const int idx = i + j*width;
  out[idx] = tex2D<float>(tex, u, v);
}

TEST_CASE("cuda_texture_buffer_upload_download_vector",
          "[cuda][texture_buffer][upload][download][vector]") {
  const std::vector<float>       texdata{1, 2, 3, 4};
  cuda::texture_buffer<float, 1, 2> tex(texdata, 2, 2);
  cuda::global_buffer<float> fetched(texdata.size());
  test_kernel0<<<dim3(2, 2), dim3(1, 1)>>>(fetched.device_ptr(),
                                           tex.device_ptr(), 2, 2);
  for (auto x : fetched.download()) { std::cerr << x << ' '; }
  std::cerr << '\n';
  auto download = fetched.download();
  for (size_t i = 0; i < texdata.size(); ++i) {
    REQUIRE(download[i] == texdata[i]);
  }
}

//==============================================================================
}  // namespace tatooine
}  // namespace gpu
}  // namespace test
//==============================================================================
