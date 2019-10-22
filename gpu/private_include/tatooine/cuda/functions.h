#ifndef TATOOINE_GPU_CUDA_FUNCTIONS_H
#define TATOOINE_GPU_CUDA_FUNCTIONS_H

#include <array>
#include "channel_format_description.h"

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================
inline void check(cudaError_t err) {
  if (err != cudaSuccess) { throw std::runtime_error{cudaGetErrorName(err)}; }
}

//==============================================================================
template <typename T, size_t NumChannels>
auto malloc_array(size_t width) {
  cudaArray_t array;
  auto desc = channel_format_description<T, NumChannels>();
  check(cudaMallocArray(&array, &desc, width));
  return array;
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t width, size_t height) {
  cudaArray_t array;
  auto       desc = channel_format_description<T, NumChannels>();
  check(cudaMallocArray(&array, &desc, width, height));
  return array;
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t width, size_t height, size_t depth) {
  cudaArray_t array;
  check(cudaMalloc3DArray(&array, channel_format_description<T, NumChannels>(), width,
                    height, depth, cudaArrayDefault));
  return array;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void malloc_array(cudaArray_t array, size_t width, size_t height, size_t depth) {
  check(cudaMalloc3DArray(&array, channel_format_description<T, NumChannels>(), width,
                    height, depth, cudaArrayDefault));
}
//==============================================================================
template <typename T, size_t NumChannels>
void memcpy_to_array(cudaArray_t dst, const std::vector<T>& src, size_t width,
                  cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
  assert(src.size() == width * NumChannels);
  check(cudaMemcpy2DToArray(
      dst, 0, 0, src.data(), NumChannels * width * sizeof(T),
      NumChannels * width * sizeof(T), 1, kind));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void memcpy_to_array(cudaArray_t dst, const std::vector<T>& src, size_t width,
                     size_t              height,
                     cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
  assert(src.size() == width * height * NumChannels);
  check(cudaMemcpy2DToArray(
      dst, 0, 0, src.data(), NumChannels * width * sizeof(T),
      NumChannels * width * sizeof(T), height, kind));
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
