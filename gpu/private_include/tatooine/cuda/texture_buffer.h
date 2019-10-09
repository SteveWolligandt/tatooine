#ifndef TATOOINE_GPU_CUDA_TEXTURE_BUFFER_H
#define TATOOINE_GPU_CUDA_TEXTURE_BUFFER_H

#include <tatooine/cxx14_type_traits.h>
#include <array>
#include <numeric>
#include <vector>
#include "functions.h"

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T, size_t NumChannels>
class texture_buffer {
 public:
  using type = T;
  static constexpr auto num_channels() { return NumChannels; }
  //============================================================================
 private:
  cudaTextureObject_t             m_texture = 0;
  std::array<size_t, NumChannels> m_resolution;
  //============================================================================
 public:
  template <typename... Resolution,
            cxx14::enable_if_arithmetic<Resolution...>...>
  texture_buffer(const std::vector<T>& host_data, Resolution... resolution)
      : m_resolution{static_cast<size_t>(resolution)...},
        m_global_buffer{num_texels()} {
    static_assert(sizeof...(Resolution) == NumChannels);
    assert(host_data.size() == num_texels());

    auto cuArray = malloc_array(resolution...);
    cudaMemcpyToArray(cuArray, 0, 0, host_data.data(), num_bytes(),
                      cudaMemcpyHostToDevice);

    // specify texture
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // specify texture object parameters
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    for (size_t i = 0; i < NumChannels; ++i) {
      texDesc.addressMode[i] = cudaAddressModeWrap;
    }
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&m_texture, &res_desc, &tex_desc, nullptr);
    cudaFreeArray(cuArray);
  }

  //----------------------------------------------------------------------------
  ~texture_buffer() { cudaDestroyTextureObject(texObj); }

  //============================================================================
 public:
  constexpr auto num_texels() {
    return std::accumulate(begin(res_arr), end(res_arr), size_t(0));
  }
  //----------------------------------------------------------------------------
  constexpr auto num_bytes() { return num_texels() * sizeof(T); }
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
