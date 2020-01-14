#ifndef TATOOINE_CUDA_TEX_CUH
#define TATOOINE_CUDA_TEX_CUH

#include <tatooine/cuda/type_traits.cuh>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <png++/png.hpp>
#include <vector>

#include <tatooine/cuda/array.cuh>
#include <tatooine/cuda/functions.cuh>
#include <tatooine/cuda/types.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T, size_t NumChannels, size_t NumDimensions>
struct tex_sampler;

template <typename T, size_t NumChannels>
struct tex_sampler<T, NumChannels, 2> {
  __device__ static auto sample(cudaTextureObject_t tex, T u, T v) {
    return tex2D<cuda::vec_t<T, NumChannels>>(tex, u, v);
  }
  __device__ static auto sample(cudaTextureObject_t      tex,
                                const cuda::vec_t<T, 2>& uv) {
    return tex2D<cuda::vec_t<T, NumChannels>>(tex, uv.x, uv.y);
  }
};

template <typename T, size_t NumChannels>
struct tex_sampler<T, NumChannels, 3> {
  __device__ static auto sample(cudaTextureObject_t tex, T u, T v, T w) {
    return tex3D<cuda::vec_t<T, NumChannels>>(tex, u, v, w);
  }
  __device__ static auto sample(cudaTextureObject_t      tex,
                                const cuda::vec_t<T, 3>& uvw) {
    return tex3D<cuda::vec_t<T, NumChannels>>(tex, uvw.x, uvw.y, uvw.z);
  }
};

enum texture_interpolation {
  point  = cudaFilterModePoint,
  linear = cudaFilterModeLinear
};

enum texture_address_mode {
  clamp  = cudaAddressModeClamp,
  wrap   = cudaAddressModeWrap,
  border = cudaAddressModeBorder,
  mirror = cudaAddressModeMirror
};

template <typename T, size_t NumChannels, size_t NumDimensions>
class tex {
 public:
  static constexpr auto num_dimensions() { return NumDimensions; }
  static constexpr auto num_channels() { return NumChannels; }
  //============================================================================
 private:
  cudaTextureObject_t                      m_device_ptr = 0;
  array<T, NumChannels, NumDimensions>     m_array;
  cuda::vec_t<unsigned int, NumDimensions> m_resolution;
  //============================================================================
 public:
  template <typename... Resolution, enable_if_integral<Resolution...> = true>
  tex(const std::vector<T>& host_data, Resolution... resolution)
      : tex{host_data, true, linear, wrap, resolution...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <typename... Resolution, enable_if_integral<Resolution...> = true>
  tex(const std::vector<T>& host_data, bool normalized_coords,
      texture_interpolation interp, texture_address_mode address_mode,
      Resolution... resolution)
      : m_array{host_data, resolution...},
        m_resolution{make_vec<unsigned int>(resolution...)} {
#ifndef NDEBUG
    std::array<size_t, sizeof...(resolution)> res_arr{resolution...};
    const size_t                              num_texels = std::accumulate(
        begin(res_arr), end(res_arr), size_t(1), std::multiplies<size_t>{});
    assert(host_data.size() == num_texels * NumChannels);
#endif
    static_assert(sizeof...(Resolution) == NumDimensions);

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));

    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = m_array.device_ptr();

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.readMode = cudaReadModeElementType;
    for (size_t i = 0; i < NumDimensions; ++i) {
      switch (address_mode) {
        case clamp: tex_desc.addressMode[i] = cudaAddressModeClamp; break;
        case wrap: tex_desc.addressMode[i] = cudaAddressModeWrap; break;
        case border: tex_desc.addressMode[i] = cudaAddressModeBorder; break;
        case mirror: tex_desc.addressMode[i] = cudaAddressModeMirror; break;
      }
    }
    switch (interp) {
      case point: tex_desc.filterMode = cudaFilterModePoint; break;
      case linear: tex_desc.filterMode = cudaFilterModeLinear; break;
    }
    tex_desc.normalizedCoords = normalized_coords;
    cudaCreateTextureObject(&m_device_ptr, &res_desc, &tex_desc, nullptr);
  }

  //----------------------------------------------------------------------------
  void free() {
    m_array.free();
    cudaDestroyTextureObject(m_device_ptr);
    m_device_ptr = 0;
  }

  //============================================================================
 public:
  __host__ __device__ auto resolution() const { return m_resolution; }
  //----------------------------------------------------------------------------
  constexpr auto device_ptr() const { return m_device_ptr; }
  //----------------------------------------------------------------------------
  constexpr const auto& array() const { return m_array; }
  constexpr auto&       array() { return m_array; }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_arithmetic<Is...> = true>
  __device__ auto sample(Is... is) const {
    static_assert(sizeof...(is) == NumDimensions);
    return tex_sampler<T, NumChannels, NumDimensions>::sample(m_device_ptr,
                                                              is...);
  }
  //----------------------------------------------------------------------------
  __device__ auto sample(const cuda::vec_t<float, NumDimensions>& uv) const {
    return tex_sampler<T, NumChannels, NumDimensions>::sample(m_device_ptr, uv);
  }
  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_arithmetic<Is...> = true>
  __device__ auto operator()(Is... is) const {
    static_assert(sizeof...(is) == NumDimensions);
    return tex_sampler<T, NumChannels, NumDimensions>::sample(m_device_ptr,
                                                              is...);
  }
  //----------------------------------------------------------------------------
  __device__ auto operator()(
      const cuda::vec_t<float, NumDimensions>& uv) const {
    return tex_sampler<T, NumChannels, NumDimensions>::sample(m_device_ptr, uv);
  }
};

template <typename T, size_t NumChannels, size_t NumDimensions>
struct is_freeable<tex<T, NumChannels, NumDimensions>> : std::true_type {};

//==============================================================================
// free functions
//==============================================================================
template <typename T, size_t NumChannels, size_t NumDimensions>
void free(tex<T, NumChannels, NumDimensions>& t) { t.free(); }

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif