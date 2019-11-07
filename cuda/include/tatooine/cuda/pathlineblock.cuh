#ifndef TATOOINE_CUDA_PATHLINE_BLOCK_CUH
#define TATOOINE_CUDA_PATHLINE_BLOCK_CUH

#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/cuda/buffer.h>
#include <tatooine/cuda/field.h>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename Real>
__global__ void pathline_block_kernel(unsteady_vectorfield<Real, 2, 2> v,
                                      vec_t<Real, 3> min, vec_t<Real, 3> max,
                                      uint3 res, size_t num_pathline_samples,
                                      buffer<Real> pathline_buffer) {
  const auto globalIdx =
      make_vec_promoted(blockIdx.x * blockDim.x + threadIdx.x,
                        blockIdx.y * blockDim.y + threadIdx.y,
                        blockIdx.z * blockDim.z + threadIdx.z);
  if (globalIdx.x >= res.x || globalIdx.y >= res.y || globalIdx.z >= res.z) {
    return;
  }
  const auto plainIdx =
      globalIdx.x + globalIdx.y * res.x + globalIdx.z * res.x * res.y;
  const auto   xt    = global_idx_to_domain_pos(globalIdx, min, max, res);
  const auto   t0Idx = plainIdx * 3 * num_pathline_samples;
  const auto   tIdx  = t0Idx + globalIdx.z;
  const size_t num_steps_forward  = num_pathline_samples - tIdx - 1;
  const size_t num_steps_backward = globalIdx.z;
}
//------------------------------------------------------------------------------
template <typename GpuReal = float, typename V, typename FieldReal,
          typename GridReal, typename SampleGridReal>
auto pathline_block(const unsteady_vectorfield<GpuReal, 2, 2>& dv,
                    const grid<GridReal, 3>& g, size_t num_pathline_samples,
                    const grid<SampleGridReal, 3>& sample_grid) {
  buffer<GpuReal> pathline_buffer(num_pathline_samples * g.num_vertices() * 3);
  auto t = static_cast<size_t>(ceil(pow(max_threads_per_block(), 1.0 / 3.0)));
  const dim3 num_threads(t, t, t);
  const dim3 num_blocks(g.dimension(0).size() / num_threads.x + 1,
                        g.dimension(1).size() / num_threads.y + 1,
                        g.dimension(2).size() / num_threads.z + 1);
  pathline_block_kernel<<<num_blocks, num_threads>>>(dv, g.min(), g.max(),
                                                     g.resolution());
  return pathline_buffer;
}
//------------------------------------------------------------------------------
template <typename GpuReal = float, typename V, typename FieldReal,
          typename GridReal, typename SampleGridReal>
auto pathline_block(const field<V, Real, 2, 2>& v, const grid<GridReal, 3>& g,
                    size_t                         num_pathline_samples,
                    const grid<SampleGridReal, 3>& sample_grid) {
  auto dv = upload<GpuReal>(v,
                            grid<SampleGridReal, 2>{sample_grid.dimension(0),
                                                    sample_grid.dimension(1)},
                            sample_grid.dimension(2));

  auto pathline_buffer = pathline_block(dv, num_pathline_samples);
  free(dv);
  return pathline_buffer;
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
