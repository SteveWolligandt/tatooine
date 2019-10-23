#ifndef TATOOINE_CUDA_SAMPLE_FIELD_CUH
#define TATOOINE_CUDA_SAMPLE_FIELD_CUH

#include <tatooine/cuda/coordinate_conversion.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

__device__ auto sample_vectorfield_steady2(cudaTextureObject_t tex, float2 pos,
                                           float2 min, float2 max, uint2 res) {
  const auto uv = domain_pos_to_uv2(pos, min, max, res);
  return tex2D<float2>(tex, uv.x, uv.y);
}

__device__ auto sample_vectorfield_unsteady2(cudaTextureObject_t tex, float2 pos,
                                           float t, float3 min, float3 max,
                                           uint3 res) {
  const auto uvw =
      domain_pos_to_uv3(make_float3(pos.x, pos.y, t), min, max, res);
  return tex3D<float2>(tex, uvw.x, uvw.y, uvw.z);
}

//==============================================================================
}  // namespace gpu
}  // namespace tatooine
//==============================================================================

#endif
