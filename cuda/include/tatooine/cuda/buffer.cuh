#ifndef TATOOINE_CUDA_BUFFER_CUH
#define TATOOINE_CUDA_BUFFER_CUH

//==============================================================================
#include <cassert>
#include <iostream>
#include <vector>
#include <tatooine/cuda/functions.cuh>
#include <tatooine/cuda/type_traits.cuh>

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
  buffer(size_t size) : m_size{size}, m_device_ptr{cuda::malloc<T>(size)} {}
  buffer(const std::vector<T>& data)
      : m_size{data.size()}, m_device_ptr{cuda::malloc<T>(data.size())} {
   cuda::memcpy(m_device_ptr, data.data(), sizeof(T) * data.size(),
               cudaMemcpyHostToDevice);
  }
  buffer(std::initializer_list<T>&& data)
      : buffer{std::vector<T>(begin(data), end(data))} {}
  //----------------------------------------------------------------------------
  void free() {
    cuda::free(m_device_ptr);
    m_device_ptr = nullptr;
  }
  //============================================================================
  // methods
  //============================================================================
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
    std::vector<T> data(m_size);
    cuda::memcpy(&data[0], m_device_ptr, sizeof(T) * m_size,
                 cudaMemcpyDeviceToHost);
    return data;
  }
  //----------------------------------------------------------------------------
  __host__ __device__ auto device_ptr() const { return m_device_ptr; }
  //----------------------------------------------------------------------------
  __host__ __device__ size_t size() const {
    return m_size;
  }
};

template <typename T>
struct is_freeable<buffer<T>> : std::true_type {};

//==============================================================================
// free functions
//==============================================================================
template <typename T>
void free(buffer<T>& b) { b.free(); }
//------------------------------------------------------------------------------
template <typename T>
void size(const buffer<T>& b) { b.size(); }

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
