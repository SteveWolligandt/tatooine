#ifndef TATOOINE_CUDA_ARRAY_CUH
#define TATOOINE_CUDA_ARRAY_CUH

#include <tatooine/cuda/functions.cuh>
#include <tatooine/utility.h>
#include <tatooine/type_traits.h>
#include <tatooine/functional.h>
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>
#include "functions.cuh"

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T, size_t NumChannels, size_t NumDimensions>
class array {
  //============================================================================
 private:
  cudaArray_t                       m_device_ptr;
  //============================================================================
 public:
  template <typename... Resolution, enable_if_arithmetic<Resolution...> = true>
  array(const std::vector<T>& host_data, Resolution... resolution)
      : m_device_ptr{malloc_array<T, NumChannels>(resolution...)} {
    static_assert(sizeof...(Resolution) == NumDimensions);
    memcpy_to_array<T, NumChannels>(m_device_ptr, host_data, resolution...);
  }
  //----------------------------------------------------------------------------
  ~array() {
#if !defined(__CUDACC__)
    cudaFreeArray(m_device_ptr);
#endif
  }
  //----------------------------------------------------------------------------
  constexpr auto device_ptr() const { return m_device_ptr; }

  auto resolution() const {
    cudaExtent  extent;
    cudaArrayGetInfo(nullptr, &extent, nullptr, m_device_ptr);
    auto res = make_array<size_t, NumDimensions>();
    res[0]   = extent.width;
    if (NumDimensions > 1) { res[1] = extent.height; }
    if (NumDimensions > 2) { res[2] = extent.depth; }
    return res;
  }
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
