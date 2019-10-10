#include <tatooine/cuda/global_buffer.h>
#include <tatooine/gpu/add.h>
#include <cassert>

//==============================================================================
namespace tatooine {
namespace gpu {
//==============================================================================

__global__ void add_kernel(int n, float *x, float *y, float *z) {
  int index  = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride) { z[i] = x[i] + y[i]; }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
std::vector<float> add(const std::vector<float> &x, const std::vector<float> &y,
                       const int block_size) {
  assert(x.size() == y.size());
  cuda::global_buffer<float> d_x(x), d_y(y), d_z(x.size());
  const int            num_blocks = (x.size() + block_size - 1) / block_size;
  add_kernel<<<num_blocks, block_size>>>(x.size(), d_x.device_ptr(),
                                         d_y.device_ptr(), d_z.device_ptr());
  cudaDeviceSynchronize();
  return d_z.download();
}

//==============================================================================
}  // namespace gpu
}  // namespace tatooine
//===================================data.data()===========================================
