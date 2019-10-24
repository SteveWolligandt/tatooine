#ifndef TATOOINE_CUDA_TYPES_CUH
#define TATOOINE_CUDA_TYPES_CUH

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T, size_t NumChannels>
struct vec_impl;

template <typename T, size_t NumChannels>
using vec = typename vec_impl<T, NumChannels>::type;

template <>
struct vec_impl<int, 1> {
  using type = int;
};
template <>
struct vec_impl<int, 2> {
  using type = int2;
};
__host__ __device__ auto make_vec(int x, int y) {
  return make_int2(x, y);
}
template <>
struct vec_impl<int, 3> {
  using type = int3;
};
__host__ __device__ auto make_vec(int x, int y, int z) {
  return make_int3(x, y, z);
}
template <>
struct vec_impl<int, 4> {
  using type = int4;
};
__host__ __device__ auto make_vec(int x, int y, int z, int w) {
  return make_int4(x, y, z, w);
}

template <>
struct vec_impl<unsigned int, 1> {
  using type = unsigned int;
};
template <>
struct vec_impl<unsigned int, 2> {
  using type = uint2;
};
__host__ __device__ auto make_vec(unsigned int x, unsigned int y) {
  return make_uint2(x, y);
}
template <>
struct vec_impl<unsigned int, 3> {
  using type = uint3;
};
  __host__ __device__ auto make_vec(unsigned int x, unsigned int y,
                                       unsigned int z) {
    return make_uint3(x, y, z);
  }
template <>
struct vec_impl<unsigned int, 4> {
  using type = uint4;
};
__host__ __device__ auto make_vec(unsigned int x, unsigned int y,
                                  unsigned int z, unsigned int w) {
  return make_uint4(x, y, z, w);
}

template <>
struct vec_impl<float, 1> {
  using type = float;
};
template <>
struct vec_impl<float, 2> {
  using type = float2;
};
__host__ __device__ auto make_vec(float x, float y) {
  return make_float2(x, y);
}
template <>
struct vec_impl<float, 3> {
  using type = float3;
};
__host__ __device__ auto make_vec(float x, float y, float z) {
  return make_float3(x, y, z);
}
template <>
struct vec_impl<float, 4> {
  using type = float4;
};
__host__ __device__ auto make_vec(float x, float y, float z, float w) {
  return make_int4(x, y, z, w);
}

template <>
struct vec_impl<double, 1> {
  using type = double;
};
template <>
struct vec_impl<double, 2> {
  using type = double2;
};
__host__ __device__ auto make_vec(double x, double y) {
  return make_double2(x, y);
}
template <>
struct vec_impl<double, 3> {
  using type = double3;
};
__host__ __device__ auto make_vec(double x, double y, double z) {
  return make_double3(x, y, z);
}
template <>
struct vec_impl<double, 4> {
  using type = double4;
};

__host__ __device__ auto make_vec(double x, double y, double z, double w) {
  return make_double4(x, y, z, w);
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
