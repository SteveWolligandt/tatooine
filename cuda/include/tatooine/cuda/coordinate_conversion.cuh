#ifndef TATOOINE_CUDA_COORDINATE_CONVERSION_CUH
#define TATOOINE_CUDA_COORDINATE_CONVERSION_CUH

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================
__device__ float2 global_idx_to_uv2(uint2 globalIdx, uint2 res) {
  // calculate normalized texture coordinates
  return make_float2(((globalIdx.x / float(res.x - 1) * (res.x * 2 - 2)) + 1) /
                         (float(res.x * 2)),
                     ((globalIdx.y / float(res.y - 1) * (res.y * 2 - 2)) + 1) /
                         (float(res.y * 2)));
}
//------------------------------------------------------------------------------
__device__ float2 global_idx_to_domain_pos2(uint2 globalIdx, float2 min,
                                           float2 max, uint2 res) {
  return make_float2(globalIdx.x / float(res.x - 1) * (max.x - min.x) + min.x,
                     globalIdx.y / float(res.y - 1) * (max.y - min.y) + min.y);
}
//------------------------------------------------------------------------------
__device__ float2 domain_pos_to_uv2(float2 domain_pos, float2 min, float2 max,
                                   uint2 res) {
  const auto grid_pos = make_float2((domain_pos.x - min.x) / (max.x - min.x),
                                    (domain_pos.y - min.y) / (max.y - min.y));
  return make_float2((grid_pos.x * (res.x * 2 - 2) + 1) / (float(res.x * 2)),
                     (grid_pos.y * (res.y * 2 - 2) + 1) / (float(res.y * 2)));
}
//==============================================================================
__device__ float3 global_idx_to_uv3(uint3 globalIdx, uint3 res) {
  // calculate normalized texture coordinates
  return make_float3(((globalIdx.x / float(res.x - 1) * (res.x * 2 - 2)) + 1) /
                         (float(res.x * 2)),
                     ((globalIdx.y / float(res.y - 1) * (res.y * 2 - 2)) + 1) /
                         (float(res.y * 2)),
                     ((globalIdx.z / float(res.z - 1) * (res.z * 2 - 2)) + 1) /
                         (float(res.z * 2)));
}
//------------------------------------------------------------------------------
__device__ float3 global_idx_to_domain_pos3(uint3 globalIdx, float3 min,
                                           float3 max, uint3 res) {
  return make_float3(globalIdx.x / float(res.x - 1) * (max.x - min.x) + min.x,
                     globalIdx.y / float(res.y - 1) * (max.y - min.y) + min.y,
                     globalIdx.z / float(res.z - 1) * (max.z - min.z) + min.z);
}
//------------------------------------------------------------------------------
__device__ float3 domain_pos_to_uv3(float3 domain_pos, float3 min, float3 max,
                                   uint3 res) {
  const auto grid_pos = make_float3((domain_pos.x - min.x) / (max.x - min.x),
                                    (domain_pos.y - min.y) / (max.y - min.y),
                                    (domain_pos.z - min.z) / (max.z - min.z));
  return make_float3((grid_pos.x * (res.x * 2 - 2) + 1) / (float(res.x * 2)),
                     (grid_pos.y * (res.y * 2 - 2) + 1) / (float(res.y * 2)),
                     (grid_pos.z * (res.z * 2 - 2) + 1) / (float(res.z * 2)));
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
