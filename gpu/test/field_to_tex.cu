#include <tatooine/cuda/global_buffer.h>
#include <tatooine/cuda/coordinate_conversion.h>
#include <tatooine/doublegyre.h>
#include <tatooine/gpu/field_to_tex.h>

#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
namespace gpu {
namespace test {
//==============================================================================
__device__ auto sample_vectorfield2(cudaTextureObject_t tex, float2 x,
                                    float2 min, float2 max, uint2 res) {
  float2 texpos = ((x - min) / (max - min)) * res;
  return tex2D<float2>(tex, texpos + 0.5);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void kernel(cudaTextureObject_t tex, float *vf_out, float *pos_out,
                       float2 min, float2 max, uint2 res) {
  const auto globalIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y);
  if (globalIdx.x >= res.x || globalIdx.y >= res.y) { return; }

  // sample vectorfield
  const auto pos = global_idx_to_domain_pos(globalIdx, min, max, res);
  const auto vf  = sample_vectorfield2();

  // sample texture and assign to output array
  const size_t plain_idx     = globalIdx.x + globalIdx.y * res.x;
  pos_out[gi * 2]            = pos.x;
  pos_out[gi * 2 + 1] = pos.y;
  vf_out[gi * 2]      = vf.x;
  vf_out[gi * 2 + 1]  = vf.y;
}

TEST_CASE("field_to_tex", "[cuda][field_to_tex][dg]") {
  // create vector field
  const numerical::doublegyre<float> v;

  // sampled vector field and upload to gpu
  const double                        t = 0;
  const linspace<double>              x_domain{0, 2, 201};
  const linspace<double>              y_domain{0, 1, 101};
  auto                       d_v = to_tex<float>(v, x_domain, y_domain, t);
  cuda::global_buffer<float> d_vf_out(2 * x_domain.size() * y_domain.size());
  cuda::global_buffer<float> d_pos_out(2 * x_domain.size() * y_domain.size());

  // call kernel
  const dim3 dimBlock{128, 128};
  const dim3 dimGrid(x_domain.size() / dimBlock.x + 1,
                     y_domain.size() / dimBlock.y + 1);
  kernel<<<dimBlock, dimGrid>>>(
      d_v.device_ptr(), d_vf_out.device_ptr(), d_pos_out.device_ptr(),
      x_domain.front(), x_domain.back(), x_domain.size(), y_domain.front(),
      y_domain.back(), y_domain.size());

  // download data from gpu
  const auto h_vf_out = d_vf_out.download();
  const auto h_pos_out = d_pos_out.download();
  for (size_t i = 0; i < h_pos_out.size(); i += 2) {
    vec<float, 2> x{h_pos_out[i], h_pos_out[i + 1]};
    vec<float, 2> v_gpu{h_vf_out[i], h_vf_out[i + 1]};
    auto v_cpu = v(x, t);
    INFO("Pos: " << x);
    INFO("CPU: " << v_cpu);
    INFO("GPU: " << v_gpu);
    REQUIRE(v_gpu(0) == Approx(v_cpu(0)).margin(1e-3));
    REQUIRE(v_gpu(1) == Approx(v_cpu(1)).margin(1e-3));
  }
}

//==============================================================================
}  // namespace test
}  // namespace gpu
}  // namespace tatooine
//==============================================================================
