#ifndef TATOOINE_CUDA_ARRAY_CUH
#define TATOOINE_CUDA_ARRAY_CUH

#include <tatooine/functional.h>
#include <tatooine/type_traits.h>

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
  ~array() {
#if !defined(__CUDACC__)
    cudaFreeArray(m_device_ptr);
#endif
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
};

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
