#ifndef TATOOINE_CUDA_FUNCTIONS_CUH
#define TATOOINE_CUDA_FUNCTIONS_CUH

#include <array>
#include <cassert>
#include <iostream>
#include <vector>

#include <tatooine/cuda/type_traits.cuh>
#include <tatooine/cuda/channel_format_description.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================
__host__ __device__ inline bool on_host() {
# ifndef __CUDA_ARCH__
  return true;
# else
  return false;
# endif
}
//------------------------------------------------------------------------------
__host__ __device__ inline bool on_device() {
# ifndef __CUDA_ARCH__
  return false;
# else
  return true;
# endif
}
//==============================================================================
inline void check(const std::string& fname, cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "[" << fname << "] - " << cudaGetErrorName(err);
  }
  assert(err == cudaSuccess);
}
//==============================================================================
template <typename T>
auto malloc(size_t elements) {
  T* device_ptr;
  check("cudaMalloc", cudaMalloc(&device_ptr, elements * sizeof(T)));
  return device_ptr;
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t width) {
  cudaArray_t array;
  auto        desc = channel_format_description<T, NumChannels>();
  check("cudaMallocArray", cudaMallocArray(&array, &desc, width));
  return array;
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t width, size_t height) {
  cudaArray_t array;
  auto        desc = channel_format_description<T, NumChannels>();
  check("cudaMallocArray", cudaMallocArray(&array, &desc, width, height));
  return array;
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array3d(size_t w, size_t h, size_t d, unsigned int flags = 0) {
  cudaArray_t array;
  auto        desc = channel_format_description<T, NumChannels>();
  cudaExtent  extent{w, h, d};
  check("cudaMalloc3DArray", cudaMalloc3DArray(&array, &desc, extent, flags));
  return array;
}
//------------------------------------------------------------------------------
inline auto malloc3d(cudaExtent extent) {
  cudaPitchedPtr pitchedDevPtr;
  check("cudaMalloc3D", cudaMalloc3D(&pitchedDevPtr, extent));
  return pitchedDevPtr;
}
//------------------------------------------------------------------------------
inline auto array_get_info(cudaChannelFormatDesc* desc, cudaExtent* extent,
                    unsigned int* flags, cudaArray_t array) {
  check("cudaArrayGetInfo",
        cudaArrayGetInfo(desc, extent, flags, array));
}
//==============================================================================
inline void memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  check("cudaMemcpy", cudaMemcpy(dst, src, count, kind));
}
//------------------------------------------------------------------------------
template <typename T>
void memcpy2d(T* dst, size_t dpitch, const T* src, size_t spitch, size_t width,
              size_t height, enum cudaMemcpyKind kind) {
  check("cudaMemcpy2D", cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind));
}
//------------------------------------------------------------------------------
inline void memcpy3d(const cudaMemcpy3DParms& p) {
  check("cudaMemcpy3D", cudaMemcpy3D(&p));
}
//==============================================================================
template <typename T, size_t NumChannels>
void memcpy_to_array(cudaArray_t dst, const std::vector<T>& src, size_t width) {
  assert(src.size() == width * NumChannels);
  check("cudaMemcpy2DToArray", cudaMemcpy2DToArray(
      dst, 0, 0, src.data(), NumChannels * width * sizeof(T),
      NumChannels * width * sizeof(T), 1, cudaMemcpyHostToDevice));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void memcpy_to_array(cudaArray_t dst, const std::vector<T>& src, size_t width,
                     size_t height) {
  assert(src.size() == width * height * NumChannels);
  check("cudaMemcpy2DToArray", cudaMemcpy2DToArray(
      dst, 0, 0, src.data(), NumChannels * width * sizeof(T),
      NumChannels * width * sizeof(T), height, cudaMemcpyHostToDevice));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void memcpy_to_array(cudaArray_t dst, const std::vector<T>& src, size_t width,
                     size_t height, size_t depth) {
  assert(src.size() == width * height * depth * NumChannels);
  check("cudaMemcpyToArray", cudaMemcpyToArray(dst, 0, 0, src.data(),
                          width * height * depth * NumChannels * sizeof(T),
                          cudaMemcpyHostToDevice));
}
//==============================================================================
template <typename T>
auto malloc_pitch(size_t width, size_t height) {
  std::pair<T*, size_t> ret;
  check("cudaMallocPitch", cudaMallocPitch(&ret.first, &ret.second, width, height));
  return ret;
}
//==============================================================================
__host__ __device__ inline void free(void* device_ptr) {
#ifdef __CUDA_ARCH__
  cudaFree(device_ptr);
#else
  check("cudaFree", cudaFree(device_ptr));
#endif
}
//------------------------------------------------------------------------------
inline void free_array(cudaArray_t device_ptr) {
  check("cudaFreeArray", cudaFreeArray(device_ptr));
}
//------------------------------------------------------------------------------
template <typename... Ts, enable_if_freeable<Ts...> = true>
void free(Ts&... ts) {
  std::array<int, sizeof...(Ts)> {(free(ts), 0)...};
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
