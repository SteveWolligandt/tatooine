#include <tatooine/doublegyre.h>
#include <tatooine/gpu/field_to_tex.h>
#include <tatooine/cuda/global_buffer.h>
#include <catch2/catch.hpp>

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
    const float u           = x / float(width - 1);
    const float v           = y / float(height - 1);
    auto        sample      = tex2D<float2>(tex, u, v);
    size_t      global_idx  = y * width + x;
    out[global_idx * 2]     = sample.x;
    out[global_idx * 2 + 1] = sample.y;
  }
}
TEST_CASE("field_to_tex",
          "[cuda][field_to_tex][dg]") {
  numerical::doublegyre<double> v;
  double                        t = 0;
  linspace<double>              x_domain{0, 2, 21};
  linspace<double>              y_domain{0, 1, 11};

  auto h_tex =
      sample_to_raw<float>(v, grid<double, 2>{x_domain, y_domain}, t);
  auto                       d_tex = to_tex<float>(v, x_domain, y_domain, t);
  cuda::global_buffer<float> d_out(x_domain.size() * y_domain.size());

  const dim3 dimBlock(16, 16);
  const dim3 dimGrid(x_domain.size() / dimBlock.x + 1, y_domain.size() / dimBlock.y + 1);
  test_kernel0<<<dimBlock, dimGrid>>>(d_tex.device_ptr(), d_out.device_ptr(),
                                      x_domain.size(), y_domain.size());
  cudaDeviceSynchronize();

  const auto h_out = d_out.download();
  for (size_t i = 0; i < h_tex.size(); ++i) {
    INFO("i = " << i);
    REQUIRE(h_out[i] == Approx(h_tex[i]).margin(1e-6));
  }
}

//==============================================================================
}  // namespace test
}  // namespace gpu
}  // namespace tatooine
//==============================================================================
