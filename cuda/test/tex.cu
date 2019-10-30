#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>
#include <tatooine/cuda/buffer.cuh>
#include <tatooine/cuda/coordinate_conversion.cuh>
#include <tatooine/cuda/tex.cuh>

#include "write_ppm.h"

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================
auto make_test_textureR(const size_t width, const size_t height,
                       const size_t depth = 1) {
  std::vector<float> h_original(width * height * depth, 0.0f);
  for (size_t z = 0; z < depth; ++z) {
    for (size_t y = 0; y < height / 2; ++y) {
      for (size_t x = 0; x < width / 2; ++x) {
        size_t i          = x + width * y + width * height * z;
        h_original[i] = 1;
      }
    }
  }
  for (size_t z = 0; z < depth; ++z) {
    for (size_t y = height / 2; y < height; ++y) {
      for (size_t x = width / 2; x < width; ++x) {
        size_t i              = x + width * y + width * height * z;
        h_original[i * 4] = 0.5f;
      }
    }
  }
  return h_original;
}
auto make_test_textureRGBA(const size_t width, const size_t height,
                       const size_t depth = 1) {
  std::vector<float> h_original(width * height * depth * 4, 0.0f);
  for (size_t z = 0; z < depth; ++z) {
    for (size_t y = 0; y < height / 2; ++y) {
      for (size_t x = 0; x < width / 2; ++x) {
        size_t i          = x + width * y + width * height * z;
        h_original[i * 4] = 1;
      }
    }
  }
  for (size_t z = 0; z < depth; ++z) {
    for (size_t y = 0; y < height / 2; ++y) {
      for (size_t x = width / 2; x < width; ++x) {
        size_t i              = x + width * y + width * height * z;
        h_original[i * 4 + 1] = 1;
      }
    }
  }
  for (size_t z = 0; z < depth; ++z) {
    for (size_t y = height / 2; y < height; ++y) {
      for (size_t x = 0; x < width / 2; ++x) {
        size_t i              = x + width * y + width * height * z;
        h_original[i * 4 + 2] = 1;
      }
    }
  }
  for (size_t z = 0; z < depth; ++z) {
    for (size_t y = height / 2; y < height; ++y) {
      for (size_t x = width / 2; x < width; ++x) {
        size_t i              = x + width * y + width * height * z;
        h_original[i * 4 + 1] = 1;
        h_original[i * 4 + 2] = 1;
      }
    }
  }
  return h_original;
}
//==============================================================================
__global__ void kernel(tex<float, 4, 2> t, buffer<float> out, float theta) {
  const auto   globalIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);
  const auto   texres    = t.resolution();
  const size_t plainIdx  = globalIdx.y * texres.x + globalIdx.x;
  if (globalIdx.x >= texres.x || globalIdx.y >= texres.y) { return; }
  // calculate normalized texture coordinates
  const auto uv = global_idx_to_uv(globalIdx, texres);

  // transform texture coordinates
  const auto tuv = make_float2(
      (uv.x - 0.5) * cosf(theta) - (uv.y - 0.5) * sinf(theta) + 0.5f,
      (uv.y - 0.5) * cosf(theta) + (uv.x - 0.5) * sinf(theta) + 0.5f);

  // sample texture and assign to output array
  const auto col        = t(tuv.x, tuv.y);
  out[plainIdx * 3]     = col.x;
  out[plainIdx * 3 + 1] = col.y;
  out[plainIdx * 3 + 2] = col.z;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
TEST_CASE("cuda_tex0", "[cuda][tex][2d][rgba][transform]") {
  const size_t width = 1024, height = 1024;
  const auto   h_original =
      make_test_textureRGBA(width, height);  // creates float-rgba texture with 4
                                             // differently colored areas
  write_ppm("untransformed.ppm", h_original, width, height, 4);

  // upload texture data to cudaArray
  tex<float, 4, 2> d_tex{h_original, width, height};

  // create device memory for output of transformed texture
  buffer<float> d_out(width * height * 3);

  // call kernel
  const dim3 numthreads(32, 32);
  const dim3 numblocks(width / numthreads.x + 1, height / numthreads.y + 1);
  kernel<<<numthreads, numblocks>>>(d_tex, d_out, M_PI / 4);

  // download transformed texture data and write
  write_ppm("transformed.ppm", d_out.download(), width, height, 3);
  free(d_tex, d_out);
}

//==============================================================================
__global__ void kernel(tex<float, 4, 2> t, buffer<float> out) {
  const auto   globalIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y);
  const auto   texres    = t.resolution();
  const size_t plainIdx  = globalIdx.y * texres.x + globalIdx.x;
  if (globalIdx.x >= texres.x || globalIdx.y >= texres.y) { return; }
  // calculate normalized texture coordinates
  const auto uv = global_idx_to_uv(globalIdx, texres);

  // sample texture and assign to output array
  const auto col        = t(uv.x, uv.y);
  out[plainIdx * 3]     = col.x;
  out[plainIdx * 3 + 1] = col.y;
  out[plainIdx * 3 + 2] = col.z;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
TEST_CASE("cuda_tex1", "[cuda][tex][2d][rgba]") {
  const size_t width = 1024, height = 1024;
  const auto   h_tex =
      make_test_textureRGBA(width, height);  // creates float-rgba texture with 4
                                         // differently colored areas

  // upload texture data
  tex<float, 4, 2> d_tex{h_tex, true, linear, border, width, height};

  // create device memory for output of transformed texture
  buffer<float> d_out(width * height * 3);

  // call kernel
  const dim3 numthreads(32, 32);
  const dim3 numblocks(width / numthreads.x + 1, height / numthreads.y + 1);
  kernel<<<numthreads, numblocks>>>(d_tex, d_out);
  auto h_out = d_out.download();
  for (size_t i = 0; i < width * height; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      REQUIRE(h_tex[i * 4 + j] == h_out[i * 3 + j]);
    }
  }
  free(d_tex, d_out);
}
//==============================================================================
__global__ void kernel(tex<float, 1, 3> t, buffer<float> out) {
  const auto globalIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y,
                                    blockIdx.z * blockDim.z + threadIdx.z);
  const auto res = t.resolution();
  if (globalIdx.x >= res.x || globalIdx.y >= res.y || globalIdx.z >= res.z) {
    return;
  }
  // sample texture and assign to output array
  const size_t plainIdx =
      globalIdx.x + globalIdx.y * res.x + globalIdx.z * res.x * res.y;
  auto uvw = global_idx_to_uvw(globalIdx, res); 
  out[plainIdx] = t(uvw);
}
TEST_CASE("cuda_tex2", "[cuda][tex][3d][r]") {
  const size_t     width = 4, height = 4, depth = 4;
  const auto       h_tex = make_test_textureR(width, height, depth);
  tex<float, 1, 3> d_tex{h_tex, true, linear, border, width, height, depth};

  // create device memory for output of transformed texture
  buffer<float> d_out(width * height * depth);

  // call kernel
  const dim3 numthreads(32, 32, 32);
  const dim3 numblocks(width  / numthreads.x + 1,
                       height / numthreads.y + 1,
                       depth  / numthreads.z + 1);
  kernel<<<numthreads, numblocks>>>(d_tex, d_out);
  auto h_out = d_out.download();
  for (size_t i = 0; i < width * height * depth; ++i) {
    INFO("i: " << i);
    REQUIRE(h_tex[i] == h_out[i]);
  }
}

//==============================================================================
__global__ void kernel(tex<float, 4, 3> t, buffer<float> out) {
  const auto globalIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y,
                                    blockIdx.z * blockDim.z + threadIdx.z);
  const auto res = t.resolution();
  if (globalIdx.x >= res.x || globalIdx.y >= res.y || globalIdx.z >= res.z) {
    return;
  }
  // sample texture and assign to output array
  const size_t plainIdx =
      globalIdx.x + globalIdx.y * res.x + globalIdx.z * res.x * res.y;
  auto uvw = global_idx_to_uvw(globalIdx, res); 
  auto col = t(uvw);
  out[plainIdx*4+0] = col.x;
  out[plainIdx*4+1] = col.y;
  out[plainIdx*4+2] = col.z;
  out[plainIdx*4+3] = col.w;
  //out[plainIdx] = uvw.x;
}
TEST_CASE("cuda_tex3", "[cuda][tex][3d][rgba]") {
  const size_t     width = 4, height = 4, depth = 4;
  const auto       h_tex = make_test_textureRGBA(width, height, depth);
  tex<float, 4, 3> d_tex{h_tex, true, linear, border, width, height, depth};

  // create device memory for output of transformed texture
  buffer<float> d_out(width * height * depth*4);

  // call kernel
  const dim3 numthreads(32, 32, 32);
  const dim3 numblocks(width  / numthreads.x + 1,
                       height / numthreads.y + 1,
                       depth  / numthreads.z + 1);
  kernel<<<numthreads, numblocks>>>(d_tex, d_out);
  auto h_out = d_out.download();
  for (size_t i = 0; i < width * height * depth; ++i) {
    INFO("i: " << i);
    for (size_t j = 0; j < 4; ++j) {
      REQUIRE(h_tex[i * 4 + j] == h_out[i * 4 + j]);
    }
  }
  free(d_tex, d_out);
}

//==============================================================================
}  // namespace test
}  // namespace cuda
}  // namespace tatooine
//==============================================================================
