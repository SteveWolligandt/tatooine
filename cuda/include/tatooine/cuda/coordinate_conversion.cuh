#ifndef TATOOINE_CUDA_COORDINATE_CONVERSION_CUH
#define TATOOINE_CUDA_COORDINATE_CONVERSION_CUH

#include <tatooine/cuda/types.cuh>
#include <tatooine/cuda/math.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================
template <typename T = float, enable_if_floating_point<T> = true>
__device__ auto global_idx_to_uv(const uint2& globalIdx, const uint2& res) {
  // calculate normalized texture coordinates
  return make_vec<T>(
      ((globalIdx.x / T(res.x - 1) * (res.x * 2 - 2)) + 1) / (T(res.x * 2)),
      ((globalIdx.y / T(res.y - 1) * (res.y * 2 - 2)) + 1) / (T(res.y * 2)));
}
//------------------------------------------------------------------------------
template <typename T = float, enable_if_floating_point<T> = true>
__device__ auto global_idx_to_uvw(const uint3& globalIdx, const uint3& res) {
  // calculate normalized texture coordinates
  return make_vec<T>(
      ((globalIdx.x / T(res.x - 1) * (res.x * 2 - 2)) + 1) / (T(res.x * 2)),
      ((globalIdx.y / T(res.y - 1) * (res.y * 2 - 2)) + 1) / (T(res.y * 2)),
      ((globalIdx.z / T(res.z - 1) * (res.z * 2 - 2)) + 1) / (T(res.z * 2)));
}
//------------------------------------------------------------------------------
template <typename T, enable_if_floating_point<T> = true>
__device__ auto global_idx_to_domain_pos(const uint2&       globalIdx,
                                         const vec_t<T, 2>& min,
                                         const vec_t<T, 2>& max,
                                         const vec_t<T, 2>& res) {
  return make_vec<T>(globalIdx.x / T(res.x - 1) * (max.x - min.x) + min.x,
                     globalIdx.y / T(res.y - 1) * (max.y - min.y) + min.y);
}
//------------------------------------------------------------------------------
template <typename T, enable_if_floating_point<T> = true>
__device__ auto domain_pos_to_uv(const vec_t<T, 2>& domain_pos,
                                 const vec_t<T, 2>& min, const vec_t<T, 2>& max,
                                 const uint2& res) {
  const auto grid_pos = make_vec<T>((domain_pos.x - min.x) / (max.x - min.x),
                                    (domain_pos.y - min.y) / (max.y - min.y));
  return make_vec<T>((grid_pos.x * (res.x * 2 - 2) + 1) / (T(res.x * 2)),
                     (grid_pos.y * (res.y * 2 - 2) + 1) / (T(res.y * 2)));
}
//==============================================================================
template <typename T = float, enable_if_floating_point<T> = true>
__device__ auto global_idx_to_domain_pos(const uint3&       globalIdx,
                                         const vec_t<T, 3>& min,
                                         const vec_t<T, 3>& max,
                                         const uint3&       res) {
  return make_vec<T>(globalIdx.x / T(res.x - 1) * (max.x - min.x) + min.x,
                     globalIdx.y / T(res.y - 1) * (max.y - min.y) + min.y,
                     globalIdx.z / T(res.z - 1) * (max.z - min.z) + min.z);
}
//------------------------------------------------------------------------------
template <typename T = float, enable_if_floating_point<T> = true>
__device__ auto domain_pos_to_uv(const vec_t<T, 3>& domain_pos,
                                 const vec_t<T, 3>& min, const vec_t<T, 3>& max,
                                 const uint3& res) {
  const auto grid_pos = make_vec<T>((domain_pos.x - min.x) / (max.x - min.x),
                                    (domain_pos.y - min.y) / (max.y - min.y),
                                    (domain_pos.z - min.z) / (max.z - min.z));
  return make_vec<T>((grid_pos.x * (res.x * 2 - 2) + 1) / (T(res.x * 2)),
                     (grid_pos.y * (res.y * 2 - 2) + 1) / (T(res.y * 2)),
                     (grid_pos.z * (res.z * 2 - 2) + 1) / (T(res.z * 2)));
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
