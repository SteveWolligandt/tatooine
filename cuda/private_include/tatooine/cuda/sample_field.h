#ifndef TATOOINE_CUDA_SAMPLE_FIELD_H
#define TATOOINE_CUDA_SAMPLE_FIELD_H

#include <tatooine/cuda/coordinate_conversion.h>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

__device__ auto sample_vectorfield_steady2(cudaTextureObject_t tex, float2 pos,
                                           float2 min, float2 max, uint2 res) {
  const auto uv = domain_pos_to_uv(pos, min, max, res);
  return tex2D<float2>(tex, uv.x, uv.y);
}

//==============================================================================
}  // namespace gpu
}  // namespace tatooine
//==============================================================================

#endif
