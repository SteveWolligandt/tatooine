#include <tatooine/cuda/global_buffer.h>
#include <tatooine/cuda/texture_buffer.h>

#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>

//==============================================================================
namespace tatooine {
namespace gpu {
namespace test {
//==============================================================================

__global__ void test_kernel0(cudatextureobject_t tex, float *out, size_t width,
                             size_t height, float theta) {
  const size_t x   = blockidx.x * blockdim.x + threadidx.x;
  const size_t y   = blockidx.y * blockdim.y + threadidx.y;
  const size_t idx = y * width + x;
  if (x < width && y < height) {
    // calculate normalized texture coordinates
    float u = x / float(width - 1);
    float v = y / float(height - 1);

    // transform texture coordinates
    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // sample texture and assign to output array
    const auto col   = tex2d<float4>(tex, tu, tu);
    out[idx * 3]     = col.x;
    out[idx * 3 + 1] = col.y;
    out[idx * 3 + 2] = col.z;
  }
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_case("cuda_texture_buffer0", "[cuda][texture_buffer]") {
  const size_t       width = 256, height = 256;
  std::vector<float> h_tex(width * height * 4, 0.0f);
  for (size_t y = 0; y < height / 2; ++y) {
    for (size_t x = 0; x < width / 2; ++x) {
      size_t i     = x + width * y;
      h_tex[i * 4] = 1;
    }
  }
  for (size_t y = 0; y < height / 2; ++y) {
    for (size_t x = width / 2; x < width; ++x) {
      size_t i         = x + width * y;
      h_tex[i * 4 + 1] = 1;
    }
  }
  for (size_t y = height / 2; y < height; ++y) {
    for (size_t x = 0; x < width / 2; ++x) {
      size_t i         = x + width * y;
      h_tex[i * 4 + 2] = 1;
    }
  }
  for (size_t y = height / 2; y < height; ++y) {
    for (size_t x = width / 2; x < width; ++x) {
      size_t i         = x + width * y;
      h_tex[i * 4 + 1] = 1;
      h_tex[i * 4 + 2] = 1;
    }
  }

  {
    std::ofstream file{"untransformed.ppm"};
    if (file.is_open()) {
      file << "p3\n" << width << ' ' << height << "\n255\n";
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          const size_t i = x + width * (height - 1 - y);
          file << static_cast<unsigned int>(h_tex[i * 4] * 255) << ' '
               << static_cast<unsigned int>(h_tex[i * 4 + 1] * 255) << ' '
               << static_cast<unsigned int>(h_tex[i * 4 + 2] * 255) << ' ';
        }
        file << '\n';
      }
    }
  }

  cuda::texture_buffer<float, 4, 2> d_tex(h_tex, width, height);
  cuda::global_buffer<float>        d_out(width * height * 3);

  const dim3 dimblock(16, 16);
  const dim3 dimgrid(width / dimblock.x + 1, height / dimblock.y + 1);
  test_kernel0<<<dimblock, dimgrid>>>(d_tex.device_ptr(), d_out.device_ptr(),
                                      width, height, m_pi / 4);
  cudadevicesynchronize();

  const auto h_out = d_out.download();
  {
    std::ofstream file{"transformed.ppm"};
    if (file.is_open()) {
      file << "p3\n" << width << ' ' << height << "\n255\n";
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          const size_t i = x + width * (height - 1 - y);
          file << static_cast<unsigned int>(h_out[i * 3] * 255) << ' '
               << static_cast<unsigned int>(h_out[i * 3 + 1] * 255) << ' '
               << static_cast<unsigned int>(h_out[i * 3 + 2] * 255) << ' ';
        }
        file << '\n';
      }
    }
  }
  for (size_t i = 0; i < h_tex.size(); ++i) {
    info("i = " << i);
    require(h_out[i] == h_tex[i]);
  }
}
//
//__global__ void test_kernel1(cudaTextureObject_t tex, float *out, size_t
//width,
//                             size_t  height) {
//  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
//  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
//  if (x < width && y < height) {
//    // calculate normalized texture coordinates
//    const float u      = x / float(width - 1);
//    const float v      = y / float(height - 1);
//    out[y * width + x] = tex2D<float>(tex, u, v);
//  }
//}
////~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TEST_CASE("cuda_texture_buffer_upload_download_vector",
//          "[cuda][texture_buffer][upload][download][vector]") {
//  const size_t       width = 10, height = 10;
//  std::vector<float> h_tex(width * height);
//  for (size_t i = 0; i < width * height; ++i) { h_tex[i] = i; }
//  cuda::texture_buffer<float, 1, 2> d_tex(h_tex, width, height);
//  cuda::global_buffer<float>        d_out(width * height);
//
//  const dim3 dimBlock(16, 16);
//  const dim3 dimGrid(width / dimBlock.x + 1, height / dimBlock.y + 1);
//  test_kernel1<<<dimBlock, dimGrid>>>(d_tex.device_ptr(), d_out.device_ptr(),
//                                      width, height);
//  cudaDeviceSynchronize();
//
//  const auto h_out = d_out.download();
//  for (size_t i = 0; i < h_tex.size(); ++i) {
//    INFO("i = " << i);
//    REQUIRE(h_out[i] == h_tex[i]);
//  }
//}
////==============================================================================
//__global__ void test_kernel2(cudaTextureObject_t tex, float *out, size_t
//width,
//                             size_t height) {
//  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
//  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
//  if (x < width && y < height) {
//    // calculate normalized texture coordinates
//    const float  u          = x / float(width - 1);
//    const float  v          = y / float(height - 1);
//    const auto   sample     = tex2D<float2>(tex, u, v);
//
//    const size_t global_idx = x + y * width;
//    out[global_idx * 2]     = sample.x;
//    out[global_idx * 2 + 1] = sample.y;
//  }
//}
////~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TEST_CASE("cuda_texture_buffer_upload_download_vector2",
//          "[cuda][texture_buffer][upload][download][vector]") {
//  const size_t       width = 10, height = 10;
//  const float        val = 5.0f;
//  std::vector<float> h_tex(width * height * 2, val);
//  for (size_t i = 0; i < width * height; ++i) { h_tex[i * 2 + 1] = i; }
//  cuda::texture_buffer<float, 2, 2> d_tex(h_tex, width, height);
//  cuda::global_buffer<float>        d_out(width * height * 2);
//
//  const dim3 dimBlock(16, 16);
//  const dim3 dimGrid(width / dimBlock.x + 1, height / dimBlock.y + 1);
//  test_kernel2<<<dimBlock, dimGrid>>>(d_tex.device_ptr(), d_out.device_ptr(),
//                                      width, height);
//  cudaDeviceSynchronize();
//
//  const auto h_out = d_out.download();
//  for (size_t i = 0; i < h_tex.size(); ++i) {
//    INFO("i = " << i);
//    REQUIRE(h_out[i] == h_tex[i]);
//  }
//}

//==============================================================================
}  // namespace test
}  // namespace gpu
}  // namespace tatooine
//==============================================================================
