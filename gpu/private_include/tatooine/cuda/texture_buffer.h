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
  //cuda::array<T, NumChannels, NumDimensions> m_array;
  cuda::global_buffer<T> m_linear;
  //============================================================================
 public:
  template <typename... Resolution,
            enable_if_arithmetic<Resolution...> = true>
  texture_buffer(const std::vector<T>& host_data, Resolution... resolution)
      //: m_array{host_data, resolution...} {
      : m_linear{host_data} {
    static_assert(sizeof...(Resolution) == NumDimensions);
    //assert(host_data.size() == num_texels() * NumChannels);

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));

    //res_desc.resType         = cudaResourceTypeArray;
    //res_desc.res.array.array = m_array.device_ptr();

    res_desc.resType           = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = m_linear.device_ptr();
    res_desc.res.linear.desc   = channel_format_description<T, NumChannels>();
    res_desc.res.linear.sizeInBytes = m_linear.num_bytes();

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.readMode = cudaReadModeElementType;
    for (size_t i = 0; i < NumDimensions; ++i) {
      tex_desc.addressMode[i] = cudaAddressModeWrap;
    }
    tex_desc.filterMode = cudaFilterModePoint;

    // sample texture and assign to output array
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = true;
    cudaCreateTextureObject(&m_device_ptr, &res_desc, &tex_desc, nullptr);
  }

  //----------------------------------------------------------------------------
  ~texture_buffer() { cudaDestroyTextureObject(m_device_ptr); }

  //============================================================================
 public:
  //constexpr auto num_texels() {
  //  return m_array.num_elements();
  //}
  //----------------------------------------------------------------------------
  //constexpr auto num_bytes() { return num_texels() * sizeof(T); }
  //----------------------------------------------------------------------------
  constexpr auto device_ptr() const { return m_device_ptr; }
  //----------------------------------------------------------------------------
  //constexpr const auto& array() const { return m_array; }
  //constexpr auto& array() { return m_array; }
  ////----------------------------------------------------------------------------
  //constexpr const auto& resolution() const { return m_array.resolution(); }
  //constexpr auto&       resolution() { return m_array.resolution(); }
  ////----------------------------------------------------------------------------
  //constexpr auto  resolution(size_t i) const { return m_array.resolution(i); }
  //constexpr auto& resolution(size_t i) { return m_array.resolution(i); }
  //----------------------------------------------------------------------------
  //constexpr auto resolution_bytes() const {
  //  auto resb = resolution();
  //  std::transform(begin(resb), end(resb), begin(resb),
  //                 [](auto r) { return r * sizeof(T) * 8; });
  //  return resb;
  //}
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
