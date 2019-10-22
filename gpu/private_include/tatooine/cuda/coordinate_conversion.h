#ifndef TATOOINE_CUDA_COORDINATE_CONVERSION_H
#define TATOOINE_CUDA_COORDINATE_CONVERSION_H

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================
__device__ float2 global_idx_to_uv(uint2 globalIdx, uint2 res) {
  // calculate normalized texture coordinates
  return make_float2(
      ((globalIdx.x / float(res.x - 1) * (res.x * 2 - 2)) + 1) /
          (float(res.x * 2)),
      ((globalIdx.y / float(res.y - 1) * (res.y * 2 - 2)) + 1) /
          (float(res.y * 2)));
}
//------------------------------------------------------------------------------
__device__ float2 global_idx_to_domain_pos(uint2 globalIdx, float2 min, float2 max, uint2 res) {
  // calculate normalized texture coordinates
  return make_float2(
      ((globalIdx.x / float(res.x - 1) * (res.x * 2 - 2)) + 1) /
          (float(res.x * 2)),
      ((globalIdx.y / float(res.y - 1) * (res.y * 2 - 2)) + 1) /
          (float(res.y * 2)));
}
//------------------------------------------------------------------------------
__device__ float2 global_idx_to_domain_pos(uint2 global_idx, float2 min,
                                            float2 max, uint2 res) {
  const auto uv = global_idx_to_uv(globalIdx, res);
  return make_float2(uv.x * (max.x - min.x) + min.x,
                     uv.y * (max.y - min.y) + min.y);
}
//------------------------------------------------------------------------------
__device__ float2 domain_pos_to_uv(float2 domain_pos, float2 min,
                                   float2 max, uint2 res) {
  const auto grid_pos = make_float2((domain_pos.x - min.x) / (max.x - min.x),
                                    (domain_pos.y - min.y) / (max.y - min.y));
  return make_float2(
      ((grid_pos.x * (res.x * 2 - 2)) + 1) / (float(res.x * 2)),
      ((grid_pos.y * (res.y * 2 - 2)) + 1) / (float(res.y * 2)));
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
