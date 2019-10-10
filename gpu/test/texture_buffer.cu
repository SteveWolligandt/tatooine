#include <tatooine/cuda/global_buffer.h>
#include <tatooine/cuda/texture_buffer.h>
#include <catch2/catch.hpp>
#include <iostream>

//==============================================================================
namespace tatooine {
namespace gpu {
namespace test {
//==============================================================================

__global__ void test_kernel0(cudaTextureObject_t tex, float *out, size_t  width,
                             size_t  height) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    // calculate normalized texture coordinates
    const float u            = x / float(width - 1);
    const float v            = y / float(height - 1);
    out[y * width + x] = tex2D<float>(tex, u, v);
  }
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TEST_CASE("cuda_texture_buffer_upload_download_vector",
          "[cuda][texture_buffer][upload][download][vector]") {
  const size_t                      width = 10, height = 10;
  const float                       val = 5.0f;
  const std::vector<float>          h_tex(width * height, val);
  cuda::texture_buffer<float, 1, 2> d_tex(h_tex, width, height);
  cuda::global_buffer<float>        d_out(width * height);

  const dim3 dimBlock(16, 16);
  const dim3 dimGrid(width / dimBlock.x + 1, height / dimBlock.y + 1);
  test_kernel0<<<dimBlock, dimGrid>>>(d_tex.device_ptr(), d_out.device_ptr(),
                                      width, height);
  cudaDeviceSynchronize();

  const auto h_out = d_out.download();
  for (size_t i = 0; i < h_tex.size(); ++i) {
    INFO("i = " << i);
    CHECK(h_out[i] == h_tex[i]);
  }
}

//==============================================================================
}  // namespace test
}  // namespace gpu
}  // namespace tatooine
//==============================================================================
