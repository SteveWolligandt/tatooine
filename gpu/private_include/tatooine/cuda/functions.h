#ifndef TATOOINE_GPU_CUDA_FUNCTIONS_H
#define TATOOINE_GPU_CUDA_FUNCTIONS_H

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
  auto       desc = channel_format_description<T, NumChannels>();
  cudaMallocArray(&array, &desc, width, height, cudaArrayDefault);
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
void memcpy_to_array(cudaArray_t dst, size_t wOffset, size_t hOffset,
                     const void* src, size_t spitch, size_t width,
                     size_t height, cudaMemcpyKind kind) {
  cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void memcpy_to_array(cudaArray_t dst, size_t wOffset, size_t hOffset,
                     const void* src, size_t spitch,
                     std::array<size_t, 2> resolution_in_bytes,
                     cudaMemcpyKind        kind) {
  cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch,
                      resolution_in_bytes[0], resolution_in_bytes[1], kind);
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
