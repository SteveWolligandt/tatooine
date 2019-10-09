#ifndef TATOOINE_GPU_CUDA_FUNCTIONS_H
#define TATOOINE_GPU_CUDA_FUNCTIONS_H

#include "channel_format_description.h"

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T, size_t NumChannels>
auto malloc_array(size_t width) {
  cudaArray* array;
  cudaMallocArray(&array, channel_format_description<T, NumChannels>(), width,
                  1, cudaArrayDefault);
  return array;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void malloc_array(cudaArray* array, size_t width) {
  cudaMallocArray(&array, channel_format_description<T, NumChannels>(), width,
                  1, cudaArrayDefault);
}
//------------------------------------------------------------------------------

template <typename T, size_t NumChannels>
auto malloc_array(size_t width, size_t height) {
  cudaArray* array;
  cudaMallocArray(&array, channel_format_description<T, NumChannels>(), width,
                  height, cudaArrayDefault);
  return array;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void malloc_array(cudaArray* array, size_t width, size_t height) {
  cudaMallocArray(&array, channel_format_description<T, NumChannels>(), width,
                  height, cudaArrayDefault);
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
void malloc_array(size_t width, size_t height, size_t depth) {
  cudaArray* array;
  cudaMalloc3DArray(&array, channel_format_description<T, NumChannels>(), width,
                    height, depth, cudaArrayDefault);
  return array;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void malloc_array(cudaArray* array, size_t width, size_t height, size_t depth) {
  cudaMalloc3DArray(&array, channel_format_description<T, NumChannels>(), width,
                    height, depth, cudaArrayDefault);
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
