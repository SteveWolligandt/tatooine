#ifndef TATOOINE_CUDA_PITCHED_MEMORY_CUH
#define TATOOINE_CUDA_PITCHED_MEMORY_CUH

//==============================================================================
#include <vector>

#include "functions.cuh"

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================
template <typename T, size_t N>
class pitched_memory;
//==============================================================================
template <typename T>
class pitched_memory<T, 2> {
  //============================================================================
  // member variables
  //============================================================================
 private:
  size_t m_width, m_height;
  T*     m_device_ptr = nullptr;
  size_t m_pitch      = 0;

  //============================================================================
  // constructors / destructor
  //============================================================================
 public:
  pitched_memory(size_t width, size_t height)
      : m_width{width}, m_height{height} {
    auto mal     = malloc_pitch<T>(width, height);
    m_device_ptr = mal.first;
    m_pitch      = mal.second;
  }
  //----------------------------------------------------------------------------
  pitched_memory(const std::vector<T>& host_data, size_t width, size_t height)
      : m_width{width}, m_height{height} {
    auto mal     = malloc_pitch<T>(width, height);
    m_device_ptr = mal.first;
    m_pitch      = mal.second;
    memcpy2d(m_device_ptr, m_pitch, host_data.data(), m_width * sizeof(T),
             m_width * sizeof(T), m_height, cudaMemcpyHostToDevice);
  }
  //----------------------------------------------------------------------------
  ~pitched_memory() {
#if !defined(__CUDACC__)
    free(m_device_ptr);
#endif
  }

  //============================================================================
  // methods
  //============================================================================
 public:
  __device__ T& at(size_t i, size_t j) {
    return *(i + reinterpret_cast<T*>(reinterpret_cast<char*>(m_device_ptr) +
                                      j * m_pitch));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ T& at(const uint2& idx) { return (*this)(idx.x, idx.y); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ T& operator()(size_t i, size_t j) { return at(i, j); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ T& operator()(const uint2& idx) { return at(idx.x, idx.y); }
  //----------------------------------------------------------------------------
  __device__ const T& at(size_t i, size_t j) const {
    return *(i + reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_device_ptr) +
                                      j * m_pitch));
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ const T& at(const uint2& idx) const {
    return (*this)(idx.x, idx.y);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ const T& operator()(size_t i, size_t j) const { return at(i, j); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ const T& operator()(const uint2& idx) const {
    return at(idx.x, idx.y);
  }
  //----------------------------------------------------------------------------
  __host__ __device__ auto width() const { return m_width; }
  __host__ __device__ auto height() const { return m_height; }
  //----------------------------------------------------------------------------
  auto download() const {
    std::vector<T> host_data(m_width * m_height);
    memcpy2d(&host_data[0], m_width * sizeof(T), m_device_ptr, m_pitch,
             m_width * sizeof(T), m_height, cudaMemcpyDeviceToHost);
    return host_data;
  }
  //----------------------------------------------------------------------------
  __host__ __device__ T const* device_ptr() const { return m_device_ptr; }
};
//==============================================================================
template <typename T>
class pitched_memory<T, 3> {
  //============================================================================
  // member variables
  //============================================================================
 private:
  size_t         m_width, m_height, m_depth;
  cudaPitchedPtr m_device_ptr;

  //============================================================================
  // constructors / destructor
  //============================================================================
 public:
  pitched_memory(size_t width, size_t height, size_t depth)
      : m_width{width},
        m_height{height},
        m_depth{depth},
        m_device_ptr{malloc3d(cudaExtent{width * sizeof(T), height, depth})} {}
  //----------------------------------------------------------------------------
  pitched_memory(const std::vector<T>& host_data, size_t width, size_t height,
                 size_t depth)
      : m_width{width},
        m_height{height},
        m_depth{depth},
        m_device_ptr{malloc3d(cudaExtent{width * sizeof(T), height, depth})} {
    cudaMemcpy3DParms p = {0};
    p.srcPtr.ptr        = const_cast<T*>(host_data.data());
    p.srcPtr.pitch      = m_width * sizeof(const T);
    p.srcPtr.xsize      = m_width;
    p.srcPtr.ysize      = m_height;
    p.dstPtr.ptr        = m_device_ptr.ptr;
    p.dstPtr.pitch      = m_device_ptr.pitch;
    p.dstPtr.xsize      = m_width;
    p.dstPtr.ysize      = m_height;
    p.extent.width      = m_width * sizeof(const T);
    p.extent.height     = m_height;
    p.extent.depth      = m_depth;
    p.kind              = cudaMemcpyHostToDevice;
    memcpy3d(p);
  }
  //----------------------------------------------------------------------------
  ~pitched_memory() {
#if !defined(__CUDACC__)
    free(m_device_ptr.ptr);
#endif
  }

  //============================================================================
  // methods
  //============================================================================
 public:
  __device__ T& at(size_t i, size_t j, size_t k) {
    const size_t slice_pitch = m_device_ptr.pitch * m_height;
    char*        slice = static_cast<char*>(m_device_ptr.ptr) + k * slice_pitch;
    T*           row   = static_cast<T*>(slice + j * m_device_ptr.pitch);
    return row[i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ T& at(const uint3& idx) { return (*this)(idx.x, idx.y); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ T& operator()(size_t i, size_t j, size_t k) { return at(i, j, k); }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ T& operator()(const uint3& idx) { return at(idx.x, idx.y, idx.z); }
  //----------------------------------------------------------------------------
  __device__ const T& at(size_t i, size_t j, size_t k) const {
    const auto slice_pitch = m_device_ptr.pitch * m_height;
    const auto slice = static_cast<char*>(m_device_ptr.ptr) + k * slice_pitch;
    const T*   row   = static_cast<T*>(slice + j * m_device_ptr.pitch);
    return row[i];
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ const T& at(const uint3& idx) const {
    return (*this)(idx.x, idx.y, idx.z);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ const T& operator()(size_t i, size_t j, size_t k) const {
    return at(i, j, k);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  __device__ const T& operator()(const uint3& idx) const {
    return at(idx.x, idx.y, idx.z);
  }
  //----------------------------------------------------------------------------
  __host__ __device__ auto width() const { return m_width; }
  __host__ __device__ auto height() const { return m_height; }
  __host__ __device__ auto depth() const { return m_depth; }
  //----------------------------------------------------------------------------
  auto download() const {
    std::vector<T> host_data(m_width * m_height * m_depth);
    cudaMemcpy3DParms p = {0};
    p.srcPtr.ptr        = m_device_ptr.ptr;
    p.srcPtr.pitch      = m_device_ptr.pitch;
    p.srcPtr.xsize      = m_width;
    p.srcPtr.ysize      = m_height;
    p.dstPtr.ptr        = const_cast<T*>(host_data.data());
    p.dstPtr.pitch      = m_width * sizeof(const T);
    p.dstPtr.xsize      = m_width;
    p.dstPtr.ysize      = m_height;
    p.extent.width      = m_width * sizeof(const T);
    p.extent.height     = m_height;
    p.extent.depth      = m_depth;
    p.kind              = cudaMemcpyDeviceToHost;
    memcpy3d(p);
    return host_data;
  }

  __host__ __device__ const auto device_ptr() const { return m_device_ptr; }
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
