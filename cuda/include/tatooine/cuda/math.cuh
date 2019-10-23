#ifndef TATOOINE_CUDA_MATH_CUH
#define TATOOINE_CUDA_MATH_CUH

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

__device__ __host__ auto operator*(const uint2& lhs, unsigned int rhs) {
  return make_uint2(lhs.x * rhs, lhs.y * rhs);
}
__device__ __host__ auto operator*(const uint3& lhs, unsigned int rhs) {
  return make_uint3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}
__device__ __host__ auto operator*(const uint4& lhs, unsigned int rhs) {
  return make_uint4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}
__device__ __host__ auto operator*(unsigned int lhs, const uint2& rhs) {
  return make_uint2(lhs * rhs.x, lhs * rhs.y);
}
__device__ __host__ auto operator*(unsigned int lhs, const uint3& rhs) {
  return make_uint3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}
__device__ __host__ auto operator*(unsigned int lhs, const uint4& rhs) {
  return make_uint4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}
__device__ __host__ auto operator*(const int2& lhs, int rhs) {
  return make_int2(lhs.x * rhs, lhs.y * rhs);
}
__device__ __host__ auto operator*(const int3& lhs, int rhs) {
  return make_int3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}
__device__ __host__ auto operator*(const int4& lhs, int rhs) {
  return make_int4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}
__device__ __host__ auto operator*(int lhs, const int2& rhs) {
  return make_int2(lhs * rhs.x, lhs * rhs.y);
}
__device__ __host__ auto operator*(int lhs, const int3& rhs) {
  return make_int3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}
__device__ __host__ auto operator*(int lhs, const int4& rhs) {
  return make_int4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}
__device__ __host__ auto operator*(const float2& lhs, float rhs) {
  return make_float2(lhs.x * rhs, lhs.y * rhs);
}
__device__ __host__ auto operator*(const float3& lhs, float rhs) {
  return make_float3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}
__device__ __host__ auto operator*(const float4& lhs, float rhs) {
  return make_float4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}
__device__ __host__ auto operator*(float lhs, const float2& rhs) {
  return make_float2(lhs * rhs.x, lhs * rhs.y);
}
__device__ __host__ auto operator*(float lhs, const float3& rhs) {
  return make_float3(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}
__device__ __host__ auto operator*(float lhs, const float4& rhs) {
  return make_float4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
}

__device__ __host__ auto operator-(const uint2& lhs, unsigned int rhs) {
  return make_uint2(lhs.x - rhs, lhs.y - rhs);
}
__device__ __host__ auto operator-(const uint3& lhs, unsigned int rhs) {
  return make_uint3(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}
__device__ __host__ auto operator-(const uint4& lhs, unsigned int rhs) {
  return make_uint4(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs);
}
__device__ __host__ auto operator-(unsigned int lhs, const uint2& rhs) {
  return make_uint2(lhs - rhs.x, lhs - rhs.y);
}
__device__ __host__ auto operator-(unsigned int lhs, const uint3& rhs) {
  return make_uint3(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}
__device__ __host__ auto operator-(unsigned int lhs, const uint4& rhs) {
  return make_uint4(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
}
__device__ __host__ auto operator-(const int2& lhs, int rhs) {
  return make_int2(lhs.x - rhs, lhs.y - rhs);
}
__device__ __host__ auto operator-(const int3& lhs, int rhs) {
  return make_int3(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}
__device__ __host__ auto operator-(const int4& lhs, int rhs) {
  return make_int4(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs);
}
__device__ __host__ auto operator-(int lhs, const int2& rhs) {
  return make_int2(lhs - rhs.x, lhs - rhs.y);
}
__device__ __host__ auto operator-(int lhs, const int3& rhs) {
  return make_int3(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}
__device__ __host__ auto operator-(int lhs, const int4& rhs) {
  return make_int4(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
}
__device__ __host__ auto operator-(const float2& lhs, float rhs) {
  return make_float2(lhs.x - rhs, lhs.y - rhs);
}
__device__ __host__ auto operator-(const float3& lhs, float rhs) {
  return make_float3(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}
__device__ __host__ auto operator-(const float4& lhs, float rhs) {
  return make_float4(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs);
}
__device__ __host__ auto operator-(float lhs, const float2& rhs) {
  return make_float2(lhs - rhs.x, lhs - rhs.y);
}
__device__ __host__ auto operator-(float lhs, const float3& rhs) {
  return make_float3(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
}
__device__ __host__ auto operator-(float lhs, const float4& rhs) {
  return make_float4(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w);
}

__device__ __host__ auto operator+(const uint2& lhs, unsigned int rhs) {
  return make_uint2(lhs.x + rhs, lhs.y + rhs);
}
__device__ __host__ auto operator+(const uint3& lhs, unsigned int rhs) {
  return make_uint3(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}
__device__ __host__ auto operator+(const uint4& lhs, unsigned int rhs) {
  return make_uint4(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs);
}
__device__ __host__ auto operator+(unsigned int lhs, const uint2& rhs) {
  return make_uint2(lhs + rhs.x, lhs + rhs.y);
}
__device__ __host__ auto operator+(unsigned int lhs, const uint3& rhs) {
  return make_uint3(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
}
__device__ __host__ auto operator+(unsigned int lhs, const uint4& rhs) {
  return make_uint4(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w);
}
__device__ __host__ auto operator+(const int2& lhs, int rhs) {
  return make_int2(lhs.x + rhs, lhs.y + rhs);
}
__device__ __host__ auto operator+(const int3& lhs, int rhs) {
  return make_int3(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}
__device__ __host__ auto operator+(const int4& lhs, int rhs) {
  return make_int4(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs);
}
__device__ __host__ auto operator+(int lhs, const int2& rhs) {
  return make_int2(lhs + rhs.x, lhs + rhs.y);
}
__device__ __host__ auto operator+(int lhs, const int3& rhs) {
  return make_int3(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
}
__device__ __host__ auto operator+(int lhs, const int4& rhs) {
  return make_int4(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w);
}
__device__ __host__ auto operator+(const float2& lhs, float rhs) {
  return make_float2(lhs.x + rhs, lhs.y + rhs);
}
__device__ __host__ auto operator+(const float3& lhs, float rhs) {
  return make_float3(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}
__device__ __host__ auto operator+(const float4& lhs, float rhs) {
  return make_float4(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs);
}
__device__ __host__ auto operator+(float lhs, const float2& rhs) {
  return make_float2(lhs + rhs.x, lhs + rhs.y);
}
__device__ __host__ auto operator+(float lhs, const float3& rhs) {
  return make_float3(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
}
__device__ __host__ auto operator+(float lhs, const float4& rhs) {
  return make_float4(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w);
}

__device__ __host__ auto operator/(const uint2& lhs, unsigned int rhs) {
  return make_uint2(lhs.x / rhs, lhs.y / rhs);
}
__device__ __host__ auto operator/(const uint3& lhs, unsigned int rhs) {
  return make_uint3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}
__device__ __host__ auto operator/(const uint4& lhs, unsigned int rhs) {
  return make_uint4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}
__device__ __host__ auto operator/(unsigned int lhs, const uint2& rhs) {
  return make_uint2(lhs / rhs.x, lhs / rhs.y);
}
__device__ __host__ auto operator/(unsigned int lhs, const uint3& rhs) {
  return make_uint3(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}
__device__ __host__ auto operator/(unsigned int lhs, const uint4& rhs) {
  return make_uint4(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}
__device__ __host__ auto operator/(const int2& lhs, int rhs) {
  return make_int2(lhs.x / rhs, lhs.y / rhs);
}
__device__ __host__ auto operator/(const int3& lhs, int rhs) {
  return make_int3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}
__device__ __host__ auto operator/(const int4& lhs, int rhs) {
  return make_int4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}
__device__ __host__ auto operator/(int lhs, const int2& rhs) {
  return make_int2(lhs / rhs.x, lhs / rhs.y);
}
__device__ __host__ auto operator/(int lhs, const int3& rhs) {
  return make_int3(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}
__device__ __host__ auto operator/(int lhs, const int4& rhs) {
  return make_int4(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}
__device__ __host__ auto operator/(const float2& lhs, float rhs) {
  return make_float2(lhs.x / rhs, lhs.y / rhs);
}
__device__ __host__ auto operator/(const float3& lhs, float rhs) {
  return make_float3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}
__device__ __host__ auto operator/(const float4& lhs, float rhs) {
  return make_float4(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}
__device__ __host__ auto operator/(float lhs, const float2& rhs) {
  return make_float2(lhs / rhs.x, lhs / rhs.y);
}
__device__ __host__ auto operator/(float lhs, const float3& rhs) {
  return make_float3(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
}
__device__ __host__ auto operator/(float lhs, const float4& rhs) {
  return make_float4(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w);
}

__device__ __host__ auto operator+(const int2& lhs, const int2& rhs) {
  return make_int2(lhs.x + rhs.x, lhs.y + rhs.y);
}
__device__ __host__ auto operator+(const int3& lhs, const int3& rhs) {
  return make_int3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
__device__ __host__ auto operator+(const int4& lhs, const int4& rhs) {
  return make_int4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}
__device__ __host__ auto operator+(const uint2& lhs, const uint2& rhs) {
  return make_uint2(lhs.x + rhs.x, lhs.y + rhs.y);
}
__device__ __host__ auto operator+(const uint3& lhs, const uint3& rhs) {
  return make_uint3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
__device__ __host__ auto operator+(const uint4& lhs, const uint4& rhs) {
  return make_uint4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}
__device__ __host__ auto operator+(const float2& lhs, const float2& rhs) {
  return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}
__device__ __host__ auto operator+(const float3& lhs, const float3& rhs) {
  return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
__device__ __host__ auto operator+(const float4& lhs, const float4& rhs) {
  return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z,
                     lhs.w + rhs.w);
}

__device__ __host__ auto operator*(const int2& lhs, const int2& rhs) {
  return make_int2(lhs.x * rhs.x, lhs.y * rhs.y);
}
__device__ __host__ auto operator*(const int3& lhs, const int3& rhs) {
  return make_int3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}
__device__ __host__ auto operator*(const int4& lhs, const int4& rhs) {
  return make_int4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}

__device__ __host__ auto operator*(const uint2& lhs, const uint2& rhs) {
  return make_uint2(lhs.x * rhs.x, lhs.y * rhs.y);
}
__device__ __host__ auto operator*(const uint3& lhs, const uint3& rhs) {
  return make_uint3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}
__device__ __host__ auto operator*(const uint4& lhs, const uint4& rhs) {
  return make_uint4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
}
__device__ __host__ auto operator*(const float2& lhs, const float2& rhs) {
  return make_float2(lhs.x * rhs.x, lhs.y * rhs.y);
}
__device__ __host__ auto operator*(const float3& lhs, const float3& rhs) {
  return make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}
__device__ __host__ auto operator*(const float4& lhs, const float4& rhs) {
  return make_float4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z,
                     lhs.w * rhs.w);
}

__device__ __host__ auto operator-(const int2& lhs, const int2& rhs) {
  return make_int2(lhs.x - rhs.x, lhs.y - rhs.y);
}
__device__ __host__ auto operator-(const int3& lhs, const int3& rhs) {
  return make_int3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
__device__ __host__ auto operator-(const int4& lhs, const int4& rhs) {
  return make_int4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

__device__ __host__ auto operator-(const uint2& lhs, const uint2& rhs) {
  return make_uint2(lhs.x - rhs.x, lhs.y - rhs.y);
}
__device__ __host__ auto operator-(const uint3& lhs, const uint3& rhs) {
  return make_uint3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
__device__ __host__ auto operator-(const uint4& lhs, const uint4& rhs) {
  return make_uint4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}
__device__ __host__ auto operator-(const float2& lhs, const float2& rhs) {
  return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}
__device__ __host__ auto operator-(const float3& lhs, const float3& rhs) {
  return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
__device__ __host__ auto operator-(const float4& lhs, const float4& rhs) {
  return make_float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z,
                     lhs.w - rhs.w);
}

__device__ __host__ auto operator/(const int2& lhs, const int2& rhs) {
  return make_int2(lhs.x / rhs.x, lhs.y / rhs.y);
}
__device__ __host__ auto operator/(const int3& lhs, const int3& rhs) {
  return make_int3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}
__device__ __host__ auto operator/(const int4& lhs, const int4& rhs) {
  return make_int4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}

__device__ __host__ auto operator/(const uint2& lhs, const uint2& rhs) {
  return make_uint2(lhs.x / rhs.x, lhs.y / rhs.y);
}
__device__ __host__ auto operator/(const uint3& lhs, const uint3& rhs) {
  return make_uint3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}
__device__ __host__ auto operator/(const uint4& lhs, const uint4& rhs) {
  return make_uint4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
}
__device__ __host__ auto operator/(const float2& lhs, const float2& rhs) {
  return make_float2(lhs.x / rhs.x, lhs.y / rhs.y);
}
__device__ __host__ auto operator/(const float3& lhs, const float3& rhs) {
  return make_float3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}
__device__ __host__ auto operator/(const float4& lhs, const float4& rhs) {
  return make_float4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z,
                     lhs.w / rhs.w);
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
