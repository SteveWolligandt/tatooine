#ifndef TATOOINE_CUDA_TYPES_CUH
#define TATOOINE_CUDA_TYPES_CUH

#include <tatooine/type_traits.h>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T, size_t NumChannels>
struct vec;

template <typename T, size_t NumChannels>
using vec_t = typename vec<T, NumChannels>::type;

template <>
struct vec<int, 1> {
  using type = int;
};
template <>
struct vec<int, 2> {
  using type = int2;
  __host__ __device__ static auto make(int x, int y) { return make_int2(x, y); }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      default:
      case 1: return v.y;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      default:
      case 1: return v.y;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};
template <>
struct vec<int, 3> {
  using type = int3;
  __host__ __device__ static auto make(int x, int y, int z) {
    return make_int3(x, y, z);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      default:
      case 2: return v.z;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      default:
      case 2: return v.z;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};
template <>
struct vec<int, 4> {
  using type = int4;
  __host__ __device__ static auto make(int x, int y, int z, int w) {
    return make_int4(x, y, z, w);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      case 2: return v.z;
      default:
      case 3: return v.w;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      case 2: return v.z;
      default:
      case 3: return v.w;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};

template <>
struct vec<unsigned int, 1> {
  using type = unsigned int;
};
template <>
struct vec<unsigned int, 2> {
  using type = uint2;
  __host__ __device__ static auto make(unsigned int x, unsigned int y) {
    return make_uint2(x, y);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      default:
      case 1: return v.y;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      default:
      case 1: return v.y;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};
template <>
struct vec<unsigned int, 3> {
  using type = uint3;
  __host__ __device__ static auto make(unsigned int x, unsigned int y,
                                       unsigned int z) {
    return make_uint3(x, y, z);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      default:
      case 2: return v.z;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      default:
      case 2: return v.z;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};
template <>
struct vec<unsigned int, 4> {
  using type = uint4;
  __host__ __device__ static auto make(unsigned int x, unsigned int y,
                                       unsigned int z, unsigned int w) {
    return make_uint4(x, y, z, w);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      case 2: return v.z;
      default:
      case 3: return v.w;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      case 2: return v.z;
      default:
      case 3: return v.w;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};

template <>
struct vec<float, 1> {
  using type = float;
};
template <>
struct vec<float, 2> {
  using type = float2;
  __host__ __device__ static auto make(float x, float y) {
    return make_float2(x, y);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      default:
      case 1: return v.y;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      default:
      case 1: return v.y;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};
template <>
struct vec<float, 3> {
  using type = float3;
  __host__ __device__ static auto make(float x, float y, float z) {
    return make_float3(x, y, z);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      default:
      case 2: return v.z;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      default:
      case 2: return v.z;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};
template <>
struct vec<float, 4> {
  using type = float4;
  __host__ __device__ static auto make(float x, float y, float z, float w) {
    return make_int4(x, y, z, w);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      case 2: return v.z;
      default:
      case 3: return v.w;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      case 2: return v.z;
      default:
      case 3: return v.w;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};

template <>
struct vec<double, 1> {
  using type = double;
};
template <>
struct vec<double, 2> {
  using type = double2;
  __host__ __device__ static auto make(double x, double y) {
    return make_double2(x, y);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      default:
      case 1: return v.y;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      default:
      case 1: return v.y;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};
template <>
struct vec<double, 3> {
  using type = double3;
  __host__ __device__ static auto make(double x, double y, double z) {
    return make_double3(x, y, z);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      default:
      case 2: return v.z;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      default:
      case 2: return v.z;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};
template <>
struct vec<double, 4> {
  using type = double4;
  __host__ __device__ static auto make(double x, double y, double z, double w) {
    return make_double4(x, y, z, w);
  }
  __host__ __device__ static auto& get(size_t i, type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      case 2: return v.z;
      default:
      case 3: return v.w;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
  __host__ __device__ static const auto& get(size_t i, const type& v) {
    switch (i) {
      case 0: return v.x;
      case 1: return v.y;
      case 2: return v.z;
      default:
      case 3: return v.w;
      //default: throw std::runtime_error{"invalid index"};
    }
  }
};

template <typename... Cs, enable_if_arithmetic<Cs...> = true>
__host__ __device__ auto make_vec_promoted(Cs... cs) {
  return cuda::vec<promote_t<Cs...>, sizeof...(Cs)>::make(cs...);
}
template <typename Out, typename... Cs, enable_if_arithmetic<Cs...> = true>
__host__ __device__ auto make_vec(Cs... cs) {
  return cuda::vec<Out, sizeof...(Cs)>::make(cs...);
}

template <size_t I, typename T, size_t N>
auto& get(vec_t<T, N>& v) {
  return vec<T, N>::get(I, v);
}
template <size_t I, typename T, size_t N>
const auto& get(const vec_t<T, N>& v) {
  return vec<T, N>::get(I, v);
}

template <typename Out, typename In, size_t N, size_t... Is>
__device__ __host__ auto make_vec(const vec_t<In, N>& v,
                                  std::index_sequence<Is...>) {
  make_vec<Out>(get<Is>(v)...);
}
template <typename Out, typename In, size_t N>
__device__ __host__ auto make_vec(const vec_t<In, N>& v) {
  return make_vec<Out>(v, std::make_index_sequence<N>{});
}
template <typename Out, typename In1, size_t N1, typename In2, size_t N2,
          size_t... Is, size_t... Js>
__device__ __host__ auto make_vec(const vec_t<In1, N1>& v1,
                                  const vec_t<In2, N2>& v2,
                                  std::index_sequence<Is...>,
                                  std::index_sequence<Js...>) {
  make_vec<Out>(get<Is>(v1)..., get<Js>(v2)...);
}
template <typename Out, typename In1, typename In2, size_t N1, size_t N2>
__device__ __host__ auto make_vec(const vec_t<In1, N1>& v1,
                                  const vec_t<In2, N2>& v2) {
  return make_vec<Out>(v1, v2, std::make_index_sequence<N1>{},
                       std::make_index_sequence<N2>{});
}

////==============================================================================
//// conversion to int
////==============================================================================
//__device__ __host__ auto make_int2(const float2& v) {
//  return make_vec<int>(v.x, v.y);
//}
//__device__ __host__ auto make_int2(const uint2& v) {
//  return make_vec<int>(v.x, v.y);
//}
//__device__ __host__ auto make_int2(const double2& v) {
//  return make_vec<int>(v.x, v.y);
//}
//__device__ __host__ auto make_int3(const float3& v) {
//  return make_vec<int>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_int3(const uint3& v) {
//  return make_vec<int>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_int3(const double3& v) {
//  return make_vec<int>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_int4(const float4& v) {
//  return make_vec<int>(v.x, v.y, v.z, v.w);
//}
//__device__ __host__ auto make_int4(const uint4& v) {
//  return make_vec<int>(v.x, v.y, v.z, v.w);
//}
//__device__ __host__ auto make_int4(const double4& v) {
//  return make_vec<int>(v.x, v.y, v.z, v.w);
//}
////==============================================================================
//// conversion to unsigned int
////==============================================================================
//__device__ __host__ auto make_uint2(const float2& v) {
//  return make_vec<unsigned int>(v.x, v.y);
//}
//__device__ __host__ auto make_uint2(const int2& v) {
//  return make_vec<unsigned int>(v.x, v.y);
//}
//__device__ __host__ auto make_uint2(const double2& v) {
//  return make_vec<unsigned int>(v.x, v.y);
//}
//__device__ __host__ auto make_uint3(const float3& v) {
//  return make_vec<unsigned int>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_uint3(const int3& v) {
//  return make_vec<unsigned int>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_uint3(const double3& v) {
//  return make_vec<unsigned int>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_uint4(const float4& v) {
//  return make_vec<unsigned int>(v.x, v.y, v.z, v.w);
//}
//__device__ __host__ auto make_uint4(const int4& v) {
//  return make_vec<unsigned int>(v.x, v.y, v.z, v.w);
//}
//__device__ __host__ auto make_uint4(const double4& v) {
//  return make_vec<unsigned int>(v.x, v.y, v.z, v.w);
//}
////==============================================================================
//// conversion to float
////==============================================================================
//__device__ __host__ auto make_float2(const int2& v) {
//  return make_vec<float>(v.x, v.y);
//}
//__device__ __host__ auto make_float2(const uint2& v) {
//  return make_vec<float>(v.x, v.y);
//}
//__device__ __host__ auto make_float2(const double2& v) {
//  return make_vec<float>(v.x, v.y);
//}
//__device__ __host__ auto make_float3(const int3& v) {
//  return make_vec<float>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_float3(const uint3& v) {
//  return make_vec<float>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_float3(const double3& v) {
//  return make_vec<float>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_float4(const int4& v) {
//  return make_vec<float>(v.x, v.y, v.z, v.w);
//}
//__device__ __host__ auto make_float4(const uint4& v) {
//  return make_vec<float>(v.x, v.y, v.z, v.w);
//}
//__device__ __host__ auto make_float4(const double4& v) {
//  return make_vec<float>(v.x, v.y, v.z, v.w);
//}
////==============================================================================
//// conversion to double
////==============================================================================
//__device__ __host__ auto make_double2(const int2& v) {
//  return make_vec<double>(v.x, v.y);
//}
//__device__ __host__ auto make_double2(const uint2& v) {
//  return make_vec<double>(v.x, v.y);
//}
//__device__ __host__ auto make_double2(const float2& v) {
//  return make_vec<double>(v.x, v.y);
//}
//__device__ __host__ auto make_double3(const int3& v) {
//  return make_vec<double>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_double3(const uint3& v) {
//  return make_vec<double>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_double3(const float3& v) {
//  return make_vec<double>(v.x, v.y, v.z);
//}
//__device__ __host__ auto make_double4(const int4& v) {
//  return make_vec<double>(v.x, v.y, v.z, v.w);
//}
//__device__ __host__ auto make_double4(const uint4& v) {
//  return make_vec<double>(v.x, v.y, v.z, v.w);
//}
//__device__ __host__ auto make_double4(const float4& v) {
//  return make_vec<double>(v.x, v.y, v.z, v.w);
//}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
