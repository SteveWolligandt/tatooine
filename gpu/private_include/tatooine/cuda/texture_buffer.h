#ifndef TATOOINE_GPU_CUDA_TEXTURE_BUFFER_H
#define TATOOINE_GPU_CUDA_TEXTURE_BUFFER_H

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
#include <tatooine/type_traits.h>
#include "array.h"
#include "functions.h"

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

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
class texture_buffer {
 public:
  using type = T;
  static constexpr auto num_dimensions() { return NumDimensions; }
  static constexpr auto num_channels() { return NumChannels; }
  //============================================================================
 private:
  cudaTextureObject_t                        m_device_ptr = 0;
  cuda::array<T, NumChannels, NumDimensions> m_array;
  //============================================================================
 public:
  template <typename... Resolution,
            enable_if_arithmetic<Resolution...> = true>
  texture_buffer(const std::vector<T>& host_data, Resolution... resolution)
      : texture_buffer{host_data, true, linear, wrap, resolution...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Resolution, enable_if_arithmetic<Resolution...> = true>
  texture_buffer(const std::vector<T>& host_data, bool normalized_coords,
                 texture_interpolation interp, texture_address_mode address_mode, Resolution... resolution)
      : m_array{host_data, resolution...} {
#ifndef NDEBUG
    std::array<size_t, sizeof...(resolution)> res_arr{resolution...};
    const size_t num_texels = std::accumulate(
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
    switch(interp) {
      case point: tex_desc.filterMode = cudaFilterModePoint; break;
      case linear: tex_desc.filterMode = cudaFilterModeLinear; break;
    }
    tex_desc.normalizedCoords = normalized_coords;
    cudaCreateTextureObject(&m_device_ptr, &res_desc, &tex_desc, nullptr);
  }

  //----------------------------------------------------------------------------
  ~texture_buffer() { cudaDestroyTextureObject(m_device_ptr); }

  //============================================================================
 public:
  //----------------------------------------------------------------------------
  constexpr auto device_ptr() const { return m_device_ptr; }
  //----------------------------------------------------------------------------
  constexpr const auto& array() const { return m_array; }
  constexpr auto& array() { return m_array; }
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
