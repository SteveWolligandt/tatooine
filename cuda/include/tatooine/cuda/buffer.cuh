#ifndef TATOOINE_CUDA_BUFFER_CUH
#define TATOOINE_CUDA_BUFFER_CUH

//==============================================================================
#include <cassert>
#include <iostream>
#include <vector>
#include <tatooine/cuda/functions.cuh>

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
  size_t m_size;
  T*     m_device_ptr = nullptr;

  //============================================================================
  // ctors / dtor
  //============================================================================
 public:
  buffer(size_t n) : m_size{n}, m_device_ptr{malloc<T>(n)} {}
  buffer(const std::vector<T>& data)
      : m_size{data.size()}, m_device_ptr{malloc<T>(data.size())} {
    memcpy(m_device_ptr, data.data(), sizeof(T) * data.size(),
               cudaMemcpyHostToDevice);
  }
  buffer(std::initializer_list<T>&& data)
      : buffer{std::vector<T>(begin(data), end(data))} {}
  //buffer(const buffer& other) :
  //  m_size{other.m_size}, m_device_ptr{malloc<T>(m_size)} {
  //  memcpy(m_device_ptr, other.m_device_ptr,
  //         sizeof(T) * m_size, cudaMemcpyDeviceToDevice);
  //}
  //----------------------------------------------------------------------------
  __host__ __device__ ~buffer() {
//#ifndef __CUDA_ARCH__
//    _free();
//#endif
  }


  //============================================================================
  // methods
  //============================================================================
 //private:
  __host__ void _free() {
    std::cerr << m_device_ptr << '\n';
    free(m_device_ptr);
    m_device_ptr = nullptr;
  }

 public:
  __device__ auto& operator[](size_t i) {
    assert(i < m_size);
    return m_device_ptr[i];
  }
  //----------------------------------------------------------------------------
  __device__ const auto& operator[](size_t i) const {
    assert(i < m_size);
    return m_device_ptr[i];
  }
  //----------------------------------------------------------------------------
  auto download() const {
    const auto     n = size();
    std::vector<T> data(n);
    memcpy(&data[0], m_device_ptr, sizeof(T) * n, cudaMemcpyDeviceToHost);
    return data;
  }
  //----------------------------------------------------------------------------
  __host__ __device__ auto device_ptr() const { return m_device_ptr; }
  __host__ __device__ size_t size() const {
    return m_size;
  }
};
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
