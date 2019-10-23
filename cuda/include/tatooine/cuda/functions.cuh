#ifndef TATOOINE_CUDA_FUNCTIONS_CUH
#define TATOOINE_CUDA_FUNCTIONS_CUH

#include <array>
#include <vector>

#include "channel_format_description.cuh"

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
  auto        desc = channel_format_description<T, NumChannels>();
  check(cudaMallocArray(&array, &desc, width));
  return array;
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t width, size_t height) {
  cudaArray_t array;
  auto        desc = channel_format_description<T, NumChannels>();
  check(cudaMallocArray(&array, &desc, width, height));
  return array;
}
//------------------------------------------------------------------------------
template <typename T, size_t NumChannels>
auto malloc_array(size_t w, size_t h, size_t d) {
  cudaArray_t array;
  auto        desc = channel_format_description<T, NumChannels>();
  cudaExtent  res{w, h, d};
  check(cudaMalloc3DArray(&array, &desc, res));
  return array;
}
//==============================================================================
template <typename T, size_t NumChannels>
void memcpy_to_array(cudaArray_t dst, const std::vector<T>& src, size_t width) {
  assert(src.size() == width * umChannels);
  check(cudaMemcpy2DToArray(
      dst, 0, 0, src.data(), NumChannels * width * sizeof(T),
      NumChannels * width * sizeof(T), 1, cudaMemcpyHostToDevice));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void memcpy_to_array(cudaArray_t dst, const std::vector<T>& src, size_t width,
                     size_t height) {
  assert(src.size() == width * height * NumChannels);
  check(cudaMemcpy2DToArray(
      dst, 0, 0, src.data(), NumChannels * width * sizeof(T),
      NumChannels * width * sizeof(T), height, cudaMemcpyHostToDevice));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t NumChannels>
void memcpy_to_array(cudaArray_t dst, const std::vector<T>& src, size_t width,
                     size_t height, size_t depth) {
  assert(src.size() == width * height * depth * NumChannels);
  check(cudaMemcpyToArray(dst, 0, 0, src.data(),
                          width * height * depth * NumChannels * sizeof(T),
                          cudaMemcpyHostToDevice));
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
