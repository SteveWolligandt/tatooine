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

template <typename T, size_t NumChannels, size_t NumDimensions>
class texture_buffer {
 public:
  using type = T;
  static constexpr auto num_dimensions() { return NumDimensions; }
  static constexpr auto num_channels() { return NumChannels; }
  //============================================================================
 private:
  cudaTextureObject_t                        m_device_ptr = 0;
  std::array<size_t, NumDimensions>          m_resolution;
  cuda::array<T, NumChannels, NumDimensions> m_array;
  //============================================================================
 public:
  template <typename... Resolution,
            enable_if_arithmetic<Resolution...> = true>
  texture_buffer(const std::vector<T>& host_data, Resolution... resolution)
      : m_resolution{static_cast<size_t>(resolution)...},
        m_array{host_data, resolution...} {
    static_assert(sizeof...(Resolution) == NumDimensions);
    assert(host_data.size() == num_texels() * NumChannels);

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = m_array.device_ptr();
    res_desc.res.width           = m_resolution[0];
    res_desc.res.height          = m_resolution[1];

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.readMode = cudaReadModeElementType;
    for (size_t i = 0; i < NumDimensions; ++i) {
      tex_desc.addressMode[i] = cudaAddressModeClamp;
    }
    tex_desc.filterMode       = cudaFilterModePoint;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = true;
    cudaCreateTextureObject(&m_device_ptr, &res_desc, &tex_desc, nullptr);
  }

  //----------------------------------------------------------------------------
  ~texture_buffer() { cudaDestroyTextureObject(m_device_ptr); }

  //============================================================================
 public:
  constexpr auto num_texels() {
    return std::accumulate(begin(m_resolution), end(m_resolution), size_t(1),
                           std::multiplies<size_t>{});
  }
  //----------------------------------------------------------------------------
  constexpr auto num_bytes() { return num_texels() * sizeof(T); }
  //----------------------------------------------------------------------------
  constexpr auto device_ptr() const { return m_device_ptr; }
  //----------------------------------------------------------------------------
  constexpr const auto& resolution() const { return m_resolution; }
  constexpr auto&       resolution() { return m_resolution; }
  //----------------------------------------------------------------------------
  constexpr auto  resolution(size_t i) const { return m_resolution[i]; }
  constexpr auto& resolution(size_t i) { return m_resolution[i]; }
  //----------------------------------------------------------------------------
  constexpr auto resolution_bytes() const {
    auto resb = m_resolution;
    std::transform(begin(resb), end(resb), begin(resb),
                   [](auto r) { return r * sizeof(T) * 8; });
    return resb;
  }
  //----------------------------------------------------------------------------
  constexpr auto resolution_bytes(size_t i) const {
    return m_resolution[i] * sizeof(T) * 8;
  }
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
