#include <tatooine/gpu/add.h>
#include <cassert>

//==============================================================================
namespace tatooine {
namespace gpu {
//==============================================================================

__global__
void global_add(int n, float *x, float *y) {
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i+=stride) {
    x[i] = x[i] + y[i];
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void add(std::vector<float>& x, const std::vector<float>& y) {
  assert(x.size() == y.size());
  float *d_x=nullptr, *d_y=nullptr;
  const size_t size = x.size() * sizeof(float);
  cudaMalloc((void**)&d_x, size);
  cudaMalloc((void**)&d_y, size);
  cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y.data(), size, cudaMemcpyHostToDevice);

  const int blockSize = 256;
  const int numBlocks = (x.size() + blockSize - 1) / blockSize;
  global_add<<<numBlocks, blockSize>>>(x.size(), d_x, d_y);
  cudaDeviceSynchronize();
  cudaMemcpy(&x[0], d_x, size, cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
}

//==============================================================================
} // namespace gpu
} // namespace tatooine
//==============================================================================
