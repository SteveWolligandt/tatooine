#ifndef TATOOINE_CUDA_LIC_H
#define TATOOINE_CUDA_LIC_H

#include <tatooine/cuda/sample_field.h>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

__global__ float2 lic(cudaTextureObject_t tex, cudaTextureObject_t* noise, float* out,
                      unsigned int num_samples, float stepwidth, float2 pos,
                      float stepwidth, float2 min, float2 max, uint2 res) {
  const auto   globalIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y);
  const size_t plainIdx  = globalIdx.x + globalIdx.y * res.x;
  if (globalIdx.x >= res.x || globalIdx.y >= res.y) { return; }

  float        val = noise[plainIdx];
  unsigned int cnt = 1;
  const auto   pos = global_idx_to_domain_pos(globalIdx, min, max, res);

  auto pos_forward = pos;
  for (unsigned int i = 0; i < num_samples; ++i) {
    pos_forward =
        rungekutta4_step_steady2(tex, pos_forward, stepwidth, min, max, res);

    if (isnan(pos_forward.x) || isnan(pos_forward.y)) { break; }
    auto uv = domain_pos_to_uv(pos_forward, min, max, res);
    val += tex2D<float>(noise, uv.x, uv.y);
    ++cnt;
  }

  auto pos_backward = pos;
  for (unsigned int i = 0; i < num_samples; ++i) {
    pos_backward =
        rungekutta4_step_steady2(tex, pos_backward, -stepwidth, min, max, res);

    if (isnan(pos_backward.x) || isnan(pos_backward.y)) { break; }
    auto uv = domain_pos_to_uv(pos_backward, min, max, res);
    val += tex2D<float>(noise, uv.x, uv.y);
    ++cnt;
  }

  out[plainIdx] = val / cnt;
}

//==============================================================================
}  // namespace gpu
}  // namespace tatooine
//==============================================================================

#endif
