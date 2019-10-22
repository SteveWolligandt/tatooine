#ifndef TATOOINE_GPU_CUDA_ARRAY_H
#define TATOOINE_GPU_CUDA_ARRAY_H

#include <tatooine/cuda/functions.h>
#include <tatooine/type_traits.h>
#include <tatooine/functional.h>
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>
#include "functions.h"

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
  ~array() { cudaFreeArray(m_device_ptr); }
  //----------------------------------------------------------------------------
  constexpr auto device_ptr() const { return m_device_ptr; }
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
