#ifndef TATOOINE_CUDA_ARRAY_CUH
#define TATOOINE_CUDA_ARRAY_CUH

#include <tatooine/functional.h>
#include <tatooine/type_traits.h>
#include <tatooine/cuda/type_traits.cuh>

#include <algorithm>
#include <array>
#include <numeric>
#include <tatooine/cuda/functions.cuh>
#include <tatooine/cuda/pitched_memory.cuh>
#include <vector>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T, size_t NumChannels, size_t NumDimensions>
class array;

template <typename T, size_t NumChannels>
class array<T, NumChannels, 2> {
  //============================================================================
 private:
  cudaArray_t m_device_ptr;
  //============================================================================
 public:
  array(const std::vector<T>& host_data, size_t width, size_t height)
      : m_device_ptr{malloc_array<T, NumChannels>(width, height)} {
    memcpy_to_array<T, NumChannels>(m_device_ptr, host_data, width, height);
  }
  //----------------------------------------------------------------------------
  void free() {
    cudaFreeArray(m_device_ptr);
  }
  //----------------------------------------------------------------------------
  auto device_ptr() const { return m_device_ptr; }
  //----------------------------------------------------------------------------
  auto resolution() const {
    auto e = extent();
    return std::array<size_t, 2>{e.width, e.height};
  }
  //----------------------------------------------------------------------------
  auto extent() const {
    cudaExtent e;
    cudaArrayGetInfo(nullptr, &e, nullptr, m_device_ptr);
    return e;
  }
  //----------------------------------------------------------------------------
  auto channel_format_description() const {
    cudaChannelFormatDesc cfd;
    cudaArrayGetInfo(&cfd, nullptr, nullptr, m_device_ptr);
    return cfd;
  }
  auto download() const {
    auto e = extent();
    std::vector<T> host_data(e.width * e.height * NumChannels);
    cudaMemcpy2DFromArray(&host_data[0], e.width * sizeof(T) * NumChannels,
                          m_device_ptr, 0, 0, e.width * sizeof(T), e.height,
                          cudaMemcpyDeviceToHost);
    return host_data;
  }
};
//==============================================================================
template <typename T, size_t NumChannels>
class array<T, NumChannels, 3> {
  //============================================================================
 private:
  cudaArray_t m_device_ptr;
  //============================================================================
 public:
  array(const std::vector<T>& host_data, size_t width, size_t height,
        size_t depth)
      : m_device_ptr{malloc_array3d<T, NumChannels>(width, height, depth)} {
    cudaMemcpy3DParms p = {0};
    p.srcPtr.ptr        = const_cast<T*>(host_data.data());
    p.srcPtr.pitch      = width * sizeof(T);
    p.srcPtr.xsize      = width;
    p.srcPtr.ysize      = height;

    p.dstArray          = m_device_ptr;

    p.extent.width      = width;
    p.extent.height     = height;
    p.extent.depth      = depth;

    p.kind              = cudaMemcpyHostToDevice;
    memcpy3d(p);
  }
  //----------------------------------------------------------------------------
  array(const pitched_memory<T, 3>& pm)
      : m_device_ptr{malloc_array3d<T, NumChannels>(pm.width(), pm.height(), pm.depth())} {
    cudaMemcpy3DParms p = {0};
    p.srcPtr            = pm.device_ptr();
    p.dstArray          = m_device_ptr;
    p.extent.width      = pm.width();
    p.extent.height     = pm.height();
    p.extent.depth      = pm.depth();

    p.kind              = cudaMemcpyDeviceToDevice;
    memcpy3d(p);
  }
  //----------------------------------------------------------------------------
  ~array() {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    cudaFreeArray(m_device_ptr);
#endif
  }
  //----------------------------------------------------------------------------
  auto device_ptr() const { return m_device_ptr; }
  //----------------------------------------------------------------------------
  auto resolution() const {
    auto e = extent();
    return std::array<size_t, 3>{e.width, e.height, e.depth};
  }
  //----------------------------------------------------------------------------
  auto extent() const {
    cudaExtent e;
    cudaArrayGetInfo(nullptr, &e, nullptr, m_device_ptr);
    return e;
  }
  //----------------------------------------------------------------------------
  auto channel_format_description() const {
    cudaChannelFormatDesc cfd;
    cudaArrayGetInfo(&cfd, nullptr, nullptr, m_device_ptr);
    return cfd;
  }
  //----------------------------------------------------------------------------
  auto download() const {
    auto e = extent();
    std::vector<T>    host_data(e.width * e.height * e.depth * NumChannels);
    cudaMemcpy3DParms p = {0};
    p.dstPtr.ptr        = const_cast<T*>(host_data.data());
    p.dstPtr.pitch      = e.width * sizeof(T);
    p.dstPtr.xsize      = e.width;
    p.dstPtr.ysize      = e.height;

    p.srcArray          = m_device_ptr;

    p.extent.width      = e.width;
    p.extent.height     = e.height;
    p.extent.depth      = e.depth;

    p.kind              = cudaMemcpyDeviceToHost;
    memcpy3d(p);
    return host_data;
  }
};

template <typename T, size_t NumChannels, size_t NumDimensions>
struct is_freeable<array<T, NumChannels, NumDimensions>> : std::true_type {};

//==============================================================================
// free functions
//==============================================================================
template <typename T, size_t NumChannels, size_t NumDimensions>
void free(array<T, NumChannels, NumDimensions>& a) { a.free(); }

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
