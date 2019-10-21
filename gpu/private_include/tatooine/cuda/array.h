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
  std::array<size_t, NumDimensions> m_resolution;

  //============================================================================
 public:
  template <typename... Resolution, enable_if_arithmetic<Resolution...> = true>
  array(const std::vector<T>& host_data, Resolution... resolution)
      : m_device_ptr{malloc_array<T, NumChannels>(resolution...)},
        m_resolution{static_cast<size_t>(resolution)...} {
    static_assert(sizeof...(Resolution) == NumDimensions);
    check(cudaMemcpy2DToArray(
        m_device_ptr, 0, 0, static_cast<const void*>(host_data.data()),
        m_resolution[0] * sizeof(T), m_resolution[0] * sizeof(T),
        m_resolution[1], cudaMemcpyHostToDevice));

    cudaExtent  levelToSize;
    check(cudaArrayGetInfo(nullptr, &levelToSize, nullptr, m_device_ptr));
  }
  template <size_t N, typename... Resolution,
            enable_if_arithmetic<Resolution...> = true>
  array(const std::array<T, N>& host_data, Resolution... resolution)
      : m_device_ptr{malloc_array<T, NumChannels>(resolution...)},
        m_resolution{static_cast<size_t>(resolution)...} {
    static_assert(sizeof...(Resolution) == NumDimensions);
    check(cudaMemcpyToArray(
        m_device_ptr, 0, 0, static_cast<const void*>(host_data.data()),
        NumChannels * sizeof(T) * num_elements(), cudaMemcpyHostToDevice));
  }

  //----------------------------------------------------------------------------
  ~array() { cudaFreeArray(m_device_ptr); }

  //============================================================================
  auto download() {
    std::vector<T> host_data(num_elements());
    memcpy_to_array(m_device_ptr, 0, 0, host_data.data(), 0, resolution_bytes(),
                    cudaMemcpyDeviceToHost);
    return host_data;
  }

  //----------------------------------------------------------------------------
  constexpr auto device_ptr() const { return m_device_ptr; }
  //----------------------------------------------------------------------------
  constexpr auto num_elements() {
    return std::accumulate(begin(m_resolution), end(m_resolution), size_t(1),
                           std::multiplies<size_t>{});
  }
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
