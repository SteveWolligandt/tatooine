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
  auto err = cudaMallocArray(&array, &desc, width,
                  1, cudaArrayDefault);
  if (err != cudaSuccess) { throw(std::runtime_error{"cuda error"}); }
  return array;
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t width, size_t height) {
  cudaArray_t array;
  auto       desc = channel_format_description<T, NumChannels>();
  auto err = cudaMallocArray(&array, &desc, width, height);
  if (err != cudaSuccess) { throw(std::runtime_error{"cuda error"}); }
  return array;
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t width, size_t height, size_t depth) {
  cudaArray_t array;
  auto err = cudaMalloc3DArray(&array, channel_format_description<T, NumChannels>(), width,
                    height, depth, cudaArrayDefault);
  if (err != cudaSuccess) { throw(std::runtime_error{"cuda error"}); }
  return array;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void malloc_array(cudaArray_t array, size_t width, size_t height, size_t depth) {
  cudaMalloc3DArray(&array, channel_format_description<T, NumChannels>(), width,
                    height, depth, cudaArrayDefault);
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
