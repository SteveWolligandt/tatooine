#ifndef TATOOINE_GPU_CUDA_GLOBAL_BUFFER_H
#define TATOOINE_GPU_CUDA_GLOBAL_BUFFER_H

#include <array>
#include <vector>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T>
class global_buffer {
 public:
  using type = T;

  //============================================================================
 private:
  T*     m_device_ptr = nullptr;
  size_t m_size       = 0;

  //============================================================================
 public:
  global_buffer() {}
  //----------------------------------------------------------------------------
  global_buffer(size_t size) : m_size{size} { malloc(num_bytes()); }
  //----------------------------------------------------------------------------
  global_buffer(const global_buffer<T>& other) : m_size{other.size()} {
    const auto bs = num_bytes();
    malloc(bs);
    cudaMemcpy(m_device_ptr, other.m_device_ptr, bs, cudaMemcpyDeviceToDevice);
  }
  //----------------------------------------------------------------------------
  global_buffer(global_buffer<T>&& other)
      : m_device_ptr{std::exchange(other.m_device_ptr, nullptr)},
        m_size{std::exchange(other.size(), 0)} {}
  //----------------------------------------------------------------------------
  auto& operator=(const global_buffer<T>& other) {
    if (m_size != other.m_size) {
      free();
      m_size = other.m_size;
      malloc(num_bytes());
    }
    cudaMemcpy(m_device_ptr, other.m_device_ptr, num_bytes(),
               cudaMemcpyDeviceToDevice);
    return *this;
  }
  //----------------------------------------------------------------------------
  auto& operator=(global_buffer<T>&& other) {
    std::swap(m_device_ptr, other.m_device_ptr);
    std::swap(m_size, other.m_size);
    return *this;
  }
  //----------------------------------------------------------------------------
  global_buffer(const std::vector<T>& host_data) : m_size{host_data.size()} {
    const auto bs = num_bytes();
    malloc(bs);
    cudaMemcpy(m_device_ptr, host_data.data(), bs, cudaMemcpyHostToDevice);
  }
  //----------------------------------------------------------------------------
  global_buffer(const std::initializer_list<T>& l) : m_size{l.size()} {
    std::vector<T> host_data(begin(l), end(l));
    const auto bs = num_bytes();
    malloc(bs);
    cudaMemcpy(m_device_ptr, host_data.data(), bs, cudaMemcpyHostToDevice);
  }
  //----------------------------------------------------------------------------
  template <size_t N>
  global_buffer(const std::array<T, N>& host_data) : m_size{N} {
    const auto bs = num_bytes();
    malloc(bs);
    cudaMemcpy(m_device_ptr, host_data.data(), bs, cudaMemcpyHostToDevice);
  }
  //----------------------------------------------------------------------------
  ~global_buffer() { free(); }

  //============================================================================
 private:
  void free() {
    cudaFree(m_device_ptr);
    m_device_ptr = nullptr;
  }
  //----------------------------------------------------------------------------
  void malloc(size_t bytes) { cudaMalloc((void**)&m_device_ptr, bytes); }

  //============================================================================
 public:
  constexpr auto device_ptr() const { return m_device_ptr; }
  constexpr auto size() const { return m_size; }
  constexpr auto num_bytes() const { return sizeof(T) * m_size; }
  auto           download() const {
    std::vector<T> host_data(m_size);
    cudaMemcpy(&host_data[0], m_device_ptr, num_bytes(),
               cudaMemcpyDeviceToHost);
    return host_data;
  }
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
