#include <tatooine/cuda/global_buffer.h>
#include <tatooine/cuda/texture_buffer.h>
#include <tatooine/cuda/coordinate_conversion.h>

#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>

#include "write_ppm.h"

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================
auto make_test_texture(const size_t width, const size_t height) {
  std::vector<float> h_original(width * height * 4, 0.0f);
  for (size_t y = 0; y < height / 2; ++y) {
    for (size_t x = 0; x < width / 2; ++x) {
      size_t i          = x + width * y;
      h_original[i * 4] = 1;
    }
  }
  for (size_t y = 0; y < height / 2; ++y) {
    for (size_t x = width / 2; x < width; ++x) {
      size_t i              = x + width * y;
      h_original[i * 4 + 1] = 1;
    }
  }
  for (size_t y = height / 2; y < height; ++y) {
    for (size_t x = 0; x < width / 2; ++x) {
      size_t i              = x + width * y;
      h_original[i * 4 + 2] = 1;
    }
  }
  for (size_t y = height / 2; y < height; ++y) {
    for (size_t x = width / 2; x < width; ++x) {
      size_t i              = x + width * y;
      h_original[i * 4 + 1] = 1;
      h_original[i * 4 + 2] = 1;
    }
  }
  return h_original;
}

__global__ void kernel0(cudaTextureObject_t tex, float* out, uint2 resolution,
                        float theta) {
  const auto   globalIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y);
  const size_t plainIdx  = globalIdx.y * resolution.x + globalIdx.x;
  if (globalIdx.x >= resolution.x || globalIdx.y >= resolution.y) { return; }
  // calculate normalized texture coordinates
  const auto uv = global_idx_to_uv(globalIdx, resolution);

  // transform texture coordinates
  const auto tuv = make_float2(
      (uv.x - 0.5) * cosf(theta) - (uv.y - 0.5) * sinf(theta) + 0.5f,
      (uv.y - 0.5) * cosf(theta) + (uv.x - 0.5) * sinf(theta) + 0.5f);

  // sample texture and assign to output array
  const auto col        = tex2D<float4>(tex, tuv.x, tuv.y);
  out[plainIdx * 3]     = col.x;
  out[plainIdx * 3 + 1] = col.y;
  out[plainIdx * 3 + 2] = col.z;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
TEST_CASE("cuda_texture_buffer0", "[cuda][texture_buffer]") {
  const size_t width = 1024, height = 1024;
  const auto   h_original =
      make_test_texture(width, height);  // creates float-rgba texture with 4
                                         // differently colored areas
  write_ppm("untransformed.ppm", h_original, width, height, 4);

  // upload texture data to cudaArray
  texture_buffer<float, 4, 2> d_tex{h_original, width, height};

  // create device memory for output of transformed texture
  global_buffer<float> d_out(width * height * 3);

  // call kernel
  const dim3 numthreads(256, 256);
  const dim3 numblocks(width / numthreads.x + 1, height / numthreads.y + 1);
  kernel0<<<numthreads, numblocks>>>(d_tex.device_ptr(), d_out.device_ptr(),
                                     make_uint2(width, height), M_PI / 4);

  // download transformed texture data and write
  write_ppm("transformed.ppm", d_out.download(), width, height, 3);
}

//==============================================================================
__global__ void kernel1(cudaTextureObject_t tex, float* out, uint2 resolution) {
  const auto   globalIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y);
  const size_t plainIdx  = globalIdx.y * resolution.x + globalIdx.x;
  if (globalIdx.x >= resolution.x || globalIdx.y >= resolution.y) { return; }
  // calculate normalized texture coordinates
  const auto uv = global_idx_to_uv(globalIdx, resolution);

  // sample texture and assign to output array
  const auto col        = tex2D<float4>(tex, uv.x, uv.y);
  out[plainIdx * 3]     = col.x;
  out[plainIdx * 3 + 1] = col.y;
  out[plainIdx * 3 + 2] = col.z;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
TEST_CASE("cuda_texture_buffer1", "[cuda][texture_buffer]") {
  const size_t width = 1024, height = 1024;
  const auto   h_tex =
      make_test_texture(width, height);  // creates float-rgba texture with 4
                                         // differently colored areas

  // upload texture data
  texture_buffer<float, 4, 2> d_tex{h_tex, true, linear, border, width, height};

  // create device memory for output of transformed texture
  global_buffer<float> d_out(width * height * 3);

  // call kernel
  const dim3 numthreads(256, 256);
  const dim3 numblocks(width / numthreads.x + 1, height / numthreads.y + 1);
  kernel1<<<numthreads, numblocks>>>(d_tex.device_ptr(), d_out.device_ptr(),
                                     make_uint2(width, height));
  auto h_out = d_out.download();
  for (size_t i = 0; i < width * height; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      REQUIRE(h_tex[i * 4 + j] == h_out[i * 3 + j]);
    }
  }
}

//==============================================================================
}  // namespace test
}  // namespace cuda
}  // namespace tatooine
//==============================================================================
