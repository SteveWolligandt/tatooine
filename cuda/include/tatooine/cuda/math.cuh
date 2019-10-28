#ifndef TATOOINE_CUDA_MATH_CUH
#define TATOOINE_CUDA_MATH_CUH

#include <tatooine/cuda/types.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

inline __device__ __host__ auto operator*(const uint2& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x * rhs, lhs.y * rhs);
}
inline __device__ __host__ auto operator*(const uint3& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}
inline __device__ __host__ auto operator*(const uint4& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}
inline __device__ __host__ auto operator*(unsigned int lhs, const uint2& rhs) {
  return make_vec<unsigned int>(lhs * rhs.x, lhs * rhs.y);
}
inline __device__ __host__ auto operator*(unsigned int lhs, const uint3& rhs) {
  return make_vec<unsigned int>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}
inline __device__ __host__ auto operator*(unsigned int lhs, const uint4& rhs) {
  return make_vec<unsigned int>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}
inline __device__ __host__ auto operator*(const int2& lhs, int rhs) {
  return make_vec<int>(lhs.x * rhs, lhs.y * rhs);
}
inline __device__ __host__ auto operator*(const int3& lhs, int rhs) {
  return make_vec<int>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}
inline __device__ __host__ auto operator*(const int4& lhs, int rhs) {
  return make_vec<int>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}
inline __device__ __host__ auto operator*(int lhs, const int2& rhs) {
  return make_vec<int>(lhs * rhs.x, lhs * rhs.y);
}
inline __device__ __host__ auto operator*(int lhs, const int3& rhs) {
  return make_vec<int>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}
inline __device__ __host__ auto operator*(int lhs, const int4& rhs) {
  return make_vec<int>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}
inline __device__ __host__ auto operator*(const float2& lhs, float rhs) {
  return make_vec<float>(lhs.x * rhs, lhs.y * rhs);
}
inline __device__ __host__ auto operator*(const float3& lhs, float rhs) {
  return make_vec<float>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}
inline __device__ __host__ auto operator*(const float4& lhs, float rhs) {
  return make_vec<float>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}
inline __device__ __host__ auto operator*(float lhs, const float2& rhs) {
  return make_vec<float>(lhs * rhs.x, lhs * rhs.y);
}
inline __device__ __host__ auto operator*(float lhs, const float3& rhs) {
  return make_vec<float>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}
inline __device__ __host__ auto operator*(float lhs, const float4& rhs) {
  return make_vec<float>(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}

inline __device__ __host__ auto operator-(const uint2& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x - rhs, lhs.y - rhs);
}
inline __device__ __host__ auto operator-(const uint3& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}
inline __device__ __host__ auto operator-(const uint4& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs);
}
inline __device__ __host__ auto operator-(unsigned int lhs, const uint2& rhs) {
  return make_vec<unsigned int>(lhs - rhs.x, lhs - rhs.y);
}
inline __device__ __host__ auto operator-(unsigned int lhs, const uint3& rhs) {
  return make_vec<unsigned int>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}
inline __device__ __host__ auto operator-(unsigned int lhs, const uint4& rhs) {
  return make_vec<unsigned int>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
}
inline __device__ __host__ auto operator-(const int2& lhs, int rhs) {
  return make_vec<int>(lhs.x - rhs, lhs.y - rhs);
}
inline __device__ __host__ auto operator-(const int3& lhs, int rhs) {
  return make_vec<int>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}
inline __device__ __host__ auto operator-(const int4& lhs, int rhs) {
  return make_vec<int>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs);
}
inline __device__ __host__ auto operator-(int lhs, const int2& rhs) {
  return make_vec<int>(lhs - rhs.x, lhs - rhs.y);
}
inline __device__ __host__ auto operator-(int lhs, const int3& rhs) {
  return make_vec<int>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}
inline __device__ __host__ auto operator-(int lhs, const int4& rhs) {
  return make_vec<int>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
}
inline __device__ __host__ auto operator-(const float2& lhs, float rhs) {
  return make_vec<float>(lhs.x - rhs, lhs.y - rhs);
}
inline __device__ __host__ auto operator-(const float3& lhs, float rhs) {
  return make_vec<float>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}
inline __device__ __host__ auto operator-(const float4& lhs, float rhs) {
  return make_vec<float>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs);
}
inline __device__ __host__ auto operator-(float lhs, const float2& rhs) {
  return make_vec<float>(lhs - rhs.x, lhs - rhs.y);
}
inline __device__ __host__ auto operator-(float lhs, const float3& rhs) {
  return make_vec<float>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}
inline __device__ __host__ auto operator-(float lhs, const float4& rhs) {
  return make_vec<float>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
}

inline __device__ __host__ auto operator+(const uint2& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x + rhs, lhs.y + rhs);
}
inline __device__ __host__ auto operator+(const uint3& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}
inline __device__ __host__ auto operator+(const uint4& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs);
}
inline __device__ __host__ auto operator+(unsigned int lhs, const uint2& rhs) {
  return make_vec<unsigned int>(lhs + rhs.x, lhs + rhs.y);
}
inline __device__ __host__ auto operator+(unsigned int lhs, const uint3& rhs) {
  return make_vec<unsigned int>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
}
inline __device__ __host__ auto operator+(unsigned int lhs, const uint4& rhs) {
  return make_vec<unsigned int>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w);
}
inline __device__ __host__ auto operator+(const int2& lhs, int rhs) {
  return make_vec<int>(lhs.x + rhs, lhs.y + rhs);
}
inline __device__ __host__ auto operator+(const int3& lhs, int rhs) {
  return make_vec<int>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}
inline __device__ __host__ auto operator+(const int4& lhs, int rhs) {
  return make_vec<int>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs);
}
inline __device__ __host__ auto operator+(int lhs, const int2& rhs) {
  return make_vec<int>(lhs + rhs.x, lhs + rhs.y);
}
inline __device__ __host__ auto operator+(int lhs, const int3& rhs) {
  return make_vec<int>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
}
inline __device__ __host__ auto operator+(int lhs, const int4& rhs) {
  return make_vec<int>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w);
}
inline __device__ __host__ auto operator+(const float2& lhs, float rhs) {
  return make_vec<float>(lhs.x + rhs, lhs.y + rhs);
}
inline __device__ __host__ auto operator+(const float3& lhs, float rhs) {
  return make_vec<float>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}
inline __device__ __host__ auto operator+(const float4& lhs, float rhs) {
  return make_vec<float>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs);
}
inline __device__ __host__ auto operator+(float lhs, const float2& rhs) {
  return make_vec<float>(lhs + rhs.x, lhs + rhs.y);
}
inline __device__ __host__ auto operator+(float lhs, const float3& rhs) {
  return make_vec<float>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
}
inline __device__ __host__ auto operator+(float lhs, const float4& rhs) {
  return make_vec<float>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w);
}

inline __device__ __host__ auto operator/(const uint2& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x / rhs, lhs.y / rhs);
}
inline __device__ __host__ auto operator/(const uint3& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}
inline __device__ __host__ auto operator/(const uint4& lhs, unsigned int rhs) {
  return make_vec<unsigned int>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}
inline __device__ __host__ auto operator/(unsigned int lhs, const uint2& rhs) {
  return make_vec<unsigned int>(lhs / rhs.x, lhs / rhs.y);
}
inline __device__ __host__ auto operator/(unsigned int lhs, const uint3& rhs) {
  return make_vec<unsigned int>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}
inline __device__ __host__ auto operator/(unsigned int lhs, const uint4& rhs) {
  return make_vec<unsigned int>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}
inline __device__ __host__ auto operator/(const int2& lhs, int rhs) {
  return make_vec<int>(lhs.x / rhs, lhs.y / rhs);
}
inline __device__ __host__ auto operator/(const int3& lhs, int rhs) {
  return make_vec<int>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}
inline __device__ __host__ auto operator/(const int4& lhs, int rhs) {
  return make_vec<int>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}
inline __device__ __host__ auto operator/(int lhs, const int2& rhs) {
  return make_vec<int>(lhs / rhs.x, lhs / rhs.y);
}
inline __device__ __host__ auto operator/(int lhs, const int3& rhs) {
  return make_vec<int>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}
inline __device__ __host__ auto operator/(int lhs, const int4& rhs) {
  return make_vec<int>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}
inline __device__ __host__ auto operator/(const float2& lhs, float rhs) {
  return make_vec<float>(lhs.x / rhs, lhs.y / rhs);
}
inline __device__ __host__ auto operator/(const float3& lhs, float rhs) {
  return make_vec<float>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}
inline __device__ __host__ auto operator/(const float4& lhs, float rhs) {
  return make_vec<float>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}
inline __device__ __host__ auto operator/(float lhs, const float2& rhs) {
  return make_vec<float>(lhs / rhs.x, lhs / rhs.y);
}
inline __device__ __host__ auto operator/(float lhs, const float3& rhs) {
  return make_vec<float>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}
inline __device__ __host__ auto operator/(float lhs, const float4& rhs) {
  return make_vec<float>(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}

inline __device__ __host__ auto operator+(const int2& lhs, const int2& rhs) {
  return make_vec<int>(lhs.x + rhs.x, lhs.y + rhs.y);
}
inline __device__ __host__ auto operator+(const int3& lhs, const int3& rhs) {
  return make_vec<int>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
inline __device__ __host__ auto operator+(const int4& lhs, const int4& rhs) {
  return make_vec<int>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}
inline __device__ __host__ auto operator+(const uint2& lhs, const uint2& rhs) {
  return make_vec<unsigned int>(lhs.x + rhs.x, lhs.y + rhs.y);
}
inline __device__ __host__ auto operator+(const uint3& lhs, const uint3& rhs) {
  return make_vec<unsigned int>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
inline __device__ __host__ auto operator+(const uint4& lhs, const uint4& rhs) {
  return make_vec<unsigned int>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}
inline __device__ __host__ auto operator+(const float2& lhs, const float2& rhs) {
  return make_vec<float>(lhs.x + rhs.x, lhs.y + rhs.y);
}
inline __device__ __host__ auto operator+(const float3& lhs, const float3& rhs) {
  return make_vec<float>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
inline __device__ __host__ auto operator+(const float4& lhs, const float4& rhs) {
  return make_vec<float>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z,
                     lhs.w + rhs.w);
}

inline __device__ __host__ auto operator*(const int2& lhs, const int2& rhs) {
  return make_vec<int>(lhs.x * rhs.x, lhs.y * rhs.y);
}
inline __device__ __host__ auto operator*(const int3& lhs, const int3& rhs) {
  return make_vec<int>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}
inline __device__ __host__ auto operator*(const int4& lhs, const int4& rhs) {
  return make_vec<int>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}

inline __device__ __host__ auto operator*(const uint2& lhs, const uint2& rhs) {
  return make_vec<unsigned int>(lhs.x * rhs.x, lhs.y * rhs.y);
}
inline __device__ __host__ auto operator*(const uint3& lhs, const uint3& rhs) {
  return make_vec<unsigned int>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}
inline __device__ __host__ auto operator*(const uint4& lhs, const uint4& rhs) {
  return make_vec<unsigned int>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}
inline __device__ __host__ auto operator*(const float2& lhs, const float2& rhs) {
  return make_vec<float>(lhs.x * rhs.x, lhs.y * rhs.y);
}
inline __device__ __host__ auto operator*(const float3& lhs, const float3& rhs) {
  return make_vec<float>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}
inline __device__ __host__ auto operator*(const float4& lhs, const float4& rhs) {
  return make_vec<float>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z,
                     lhs.w * rhs.w);
}

inline __device__ __host__ auto operator-(const int2& lhs, const int2& rhs) {
  return make_vec<int>(lhs.x - rhs.x, lhs.y - rhs.y);
}
inline __device__ __host__ auto operator-(const int3& lhs, const int3& rhs) {
  return make_vec<int>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
inline __device__ __host__ auto operator-(const int4& lhs, const int4& rhs) {
  return make_vec<int>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

inline __device__ __host__ auto operator-(const uint2& lhs, const uint2& rhs) {
  return make_vec<unsigned int>(lhs.x - rhs.x, lhs.y - rhs.y);
}
inline __device__ __host__ auto operator-(const uint3& lhs, const uint3& rhs) {
  return make_vec<unsigned int>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
inline __device__ __host__ auto operator-(const uint4& lhs, const uint4& rhs) {
  return make_vec<unsigned int>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}
inline __device__ __host__ auto operator-(const float2& lhs, const float2& rhs) {
  return make_vec<float>(lhs.x - rhs.x, lhs.y - rhs.y);
}
inline __device__ __host__ auto operator-(const float3& lhs, const float3& rhs) {
  return make_vec<float>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
inline __device__ __host__ auto operator-(const float4& lhs, const float4& rhs) {
  return make_vec<float>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z,
                     lhs.w - rhs.w);
}

inline __device__ __host__ auto operator/(const int2& lhs, const int2& rhs) {
  return make_vec<int>(lhs.x / rhs.x, lhs.y / rhs.y);
}
inline __device__ __host__ auto operator/(const int3& lhs, const int3& rhs) {
  return make_vec<int>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}
inline __device__ __host__ auto operator/(const int4& lhs, const int4& rhs) {
  return make_vec<int>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}

inline __device__ __host__ auto operator/(const uint2& lhs, const uint2& rhs) {
  return make_vec<unsigned int>(lhs.x / rhs.x, lhs.y / rhs.y);
}
inline __device__ __host__ auto operator/(const uint3& lhs, const uint3& rhs) {
  return make_vec<unsigned int>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}
inline __device__ __host__ auto operator/(const uint4& lhs, const uint4& rhs) {
  return make_vec<unsigned int>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}
inline __device__ __host__ auto operator/(const float2& lhs, const float2& rhs) {
  return make_vec<float>(lhs.x / rhs.x, lhs.y / rhs.y);
}
inline __device__ __host__ auto operator/(const float3& lhs, const float3& rhs) {
  return make_vec<float>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}
inline __device__ __host__ auto operator/(const float4& lhs, const float4& rhs) {
  return make_vec<float>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z,
                     lhs.w / rhs.w);
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
