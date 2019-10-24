#ifndef TATOOINE_CUDA_BUFFER_CUH
#define TATOOINE_CUDA_BUFFER_CUH

//==============================================================================
#include <cassert>
#include <vector>
#include "var.cuh"

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================
template <typename T>
class buffer {
  //============================================================================
  // members
  //============================================================================
 private:
  var<size_t> m_size;
  T*          m_device_ptr = nullptr;

  //============================================================================
  // ctors / dtor
  //============================================================================
 public:
  buffer(size_t n) : m_size{n} { cudaMalloc(&m_device_ptr, n * sizeof(T)); }
  buffer(const std::vector<T>& data) : m_size{data.size()} {
    cudaMalloc(&m_device_ptr, data.size() * sizeof(T));
    cudaMemcpy(m_device_ptr, data.data(), sizeof(T) * data.size(),
               cudaMemcpyHostToDevice);
  }
  buffer(std::initializer_list<T>&& data)
      : buffer{std::vector<T>(begin(data), end(data))} {}
  //----------------------------------------------------------------------------
  __host__ __device__ ~buffer() {
#if !defined(__CUDACC__)
    cudaFree(m_device_ptr);
    m_device_ptr = nullptr;
#endif
  }
  //============================================================================
  // methods
  //============================================================================
 public:
  __device__ auto& operator[](size_t i) {
    assert(i < *m_size);
    return m_device_ptr[i];
  }
  //----------------------------------------------------------------------------
  __device__ const auto& operator[](size_t i) const {
    assert(i < *m_size);
    return m_device_ptr[i];
  }
  //----------------------------------------------------------------------------
  auto download() const {
    const auto     n = size();
    std::vector<T> data(n);
    cudaMemcpy(&data[0], m_device_ptr, sizeof(T) * n, cudaMemcpyDeviceToHost);
    return data;
  }
  //----------------------------------------------------------------------------
  __host__ __device__ auto device_ptr() const { return m_device_ptr; }
  __host__ __device__ size_t size() const {
#ifdef __CUDA_ARCH__
    return *m_size;
#else
    return m_size.download();
#endif
  }
  //----------------------------------------------------------------------------
  const auto& device_size() const { return m_size; }
};
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
