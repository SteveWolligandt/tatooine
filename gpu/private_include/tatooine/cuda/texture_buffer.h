#ifndef TATOOINE_GPU_CUDA_TEXTURE_BUFFER_H
#define TATOOINE_GPU_CUDA_TEXTURE_BUFFER_H

#include <tatooine/cxx14_type_traits.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <vector>
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
  cudaTextureObject_t               m_device_ptr = 0;
  std::array<size_t, NumDimensions> m_resolution;
  //============================================================================
 public:
  template <typename... Resolution,
            cxx14::enable_if_arithmetic<Resolution...> = true>
  texture_buffer(const std::vector<T>& host_data, Resolution... resolution)
      : m_resolution{static_cast<size_t>(resolution)...} {
    static_assert(sizeof...(Resolution) == NumDimensions);
    assert(host_data.size() == num_texels() * NumChannels);

    auto cuArray = malloc_array<T, NumChannels>(num_texels());
    cuda::memcpy_to_array(cuArray, 0, 0,
                          static_cast<const void*>(host_data.data()), 0,
                          resolution_bytes(), cudaMemcpyHostToDevice);

    // specify texture
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = cuArray;

    // specify texture object parameters
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    for (size_t i = 0; i < NumDimensions; ++i) {
      tex_desc.addressMode[i] = cudaAddressModeWrap;
    }
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    cudaCreateTextureObject(&m_device_ptr, &res_desc, &tex_desc, nullptr);
    cudaFreeArray(cuArray);
  }

  //----------------------------------------------------------------------------
  ~texture_buffer() { cudaDestroyTextureObject(m_device_ptr); }

  //============================================================================
 public:
  constexpr auto num_texels() {
    return std::accumulate(begin(m_resolution), end(m_resolution), size_t(0));
  }
  //----------------------------------------------------------------------------
  constexpr auto num_bytes() { return num_texels() * sizeof(T); }
  //----------------------------------------------------------------------------
  constexpr auto device_ptr() const { return m_device_ptr; }
  //----------------------------------------------------------------------------
  constexpr auto resolution() const { return m_resolution; }
  //----------------------------------------------------------------------------
  constexpr auto resolution(size_t i) const { return m_resolution[i]; }
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
