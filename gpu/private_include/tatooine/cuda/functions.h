#ifndef TATOOINE_GPU_CUDA_FUNCTIONS_H
#define TATOOINE_GPU_CUDA_FUNCTIONS_H

#include <array>
#include "channel_format_description.h"

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T, size_t NumChannels>
auto malloc_array(size_t width) {
  cudaArray_t array;
  auto desc = channel_format_description<T, NumChannels>();
  cudaMallocArray(&array, &desc, width,
                  1, cudaArrayDefault);
  return array;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void malloc_array(cudaArray_t array, size_t width) {
  cudaMallocArray(&array, channel_format_description<T, NumChannels>(), width,
                  1, cudaArrayDefault);
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t width, size_t height) {
  cudaArray_t array;
  //auto       desc = channel_format_description<T, NumChannels>();
  cudaChannelFormatDesc desc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaMallocArray(&array, &desc, width, height);
  return array;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void malloc_array(cudaArray_t array, size_t width, size_t height) {
  cudaMallocArray(&array, channel_format_description<T, NumChannels>(), width,
                  height, cudaArrayDefault);
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t width, size_t height, size_t depth) {
  cudaArray_t array;
  cudaMalloc3DArray(&array, channel_format_description<T, NumChannels>(), width,
                    height, depth, cudaArrayDefault);
  return array;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void malloc_array(cudaArray_t array, size_t width, size_t height, size_t depth) {
  cudaMalloc3DArray(&array, channel_format_description<T, NumChannels>(), width,
                    height, depth, cudaArrayDefault);
}

//==============================================================================
template <typename T>
void memcpy_to_array(cudaArray_t dst, size_t wOffset, size_t hOffset,
                     const T* src, size_t spitch, size_t width,
                     size_t height, cudaMemcpyKind kind) {
  // cudaMemcpy2DToArray(dst, wOffset, hOffset, src, 0,
  //                     width * sizeof(T), height * sizeof(T), kind);
  cudaMemcpyToArray(dst, 0, 0, (const void*)src, width * height * sizeof(T),
                    cudaMemcpyHostToDevice);
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
