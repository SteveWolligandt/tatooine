#ifndef TATOOINE_CUDA_PATHLINE_BLOCK_CUH
#define TATOOINE_CUDA_PATHLINE_BLOCK_CUH

#include <tatooine/cuda/buffer.cuh>
#include <tatooine/cuda/field.cuh>
#include <tatooine/cuda/rungekutta4.cuh>
#include <tatooine/field.h>
#include <tatooine/grid.h>

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

  const auto xt           = global_idx_to_domain_pos(globalIdx, min, max, res);
  const auto t0Idx        = plainIdx * 3 * num_pathline_samples;
  const auto x            = make_vec<Real>(xt.x, xt.y);
  const auto t            = xt.z;
  const auto normalized_t = (t - min.z) / (max.z - min.z);

  const auto num_steps_forward = static_cast<size_t>(
      floor((num_pathline_samples - 1) * normalized_t));
  const auto num_steps_backward = num_pathline_samples - num_steps_forward - 1;
  const auto tIdx = t0Idx + num_steps_backward * 3;
  const auto stepsize_backward  = (v.tmin() - t) / num_steps_backward;
  const auto stepsize_forward   = (v.tmax() - t) / num_steps_forward;

  // write initial position
  pathline_buffer[tIdx]     = x.x;
  pathline_buffer[tIdx + 1] = x.y;
  pathline_buffer[tIdx + 2] = t;
  
  auto cur_x = x;
  auto cur_t = t;
  for (size_t i = 0; i < num_steps_backward; ++i) {
    cur_x = rungekutta4_step(v, cur_x, cur_t, stepsize_backward);
    cur_t += stepsize_backward;
    if (abs(cur_t - min.z) < 1e-5) { cur_t = min.z; }
    pathline_buffer[tIdx - (i+1) * 3]     = cur_x.x;
    pathline_buffer[tIdx - (i+1) * 3 + 1] = cur_x.y;
    pathline_buffer[tIdx - (i+1) * 3 + 2] = cur_t;
  }
  cur_x = x;
  cur_t = t;
  for (size_t i = 0; i < num_steps_forward; ++i) {
    cur_x = rungekutta4_step(v, cur_x, cur_t, stepsize_forward);
    cur_t += stepsize_forward;
    if (abs(cur_t - max.z) < 1e-5) { cur_t = max.z; }
    pathline_buffer[tIdx + (i+1) * 3]     = cur_x.x;
    pathline_buffer[tIdx + (i+1) * 3 + 1] = cur_x.y;
    pathline_buffer[tIdx + (i+1) * 3 + 2] = cur_t;
  }
}
//------------------------------------------------------------------------------
template <typename GpuReal = float, typename GridReal>
auto pathline_block(const unsteady_vectorfield<GpuReal, 2, 2>& dv,
                    const grid<GridReal, 3>& g, size_t num_pathline_samples) {
  buffer<GpuReal> pathline_buffer(num_pathline_samples * g.num_vertices() * 3);
  const dim3 num_threads(32, 32, 1);
  const dim3 num_blocks(g.dimension(0).size() / num_threads.x + 1,
                        g.dimension(1).size() / num_threads.y + 1,
                        g.dimension(2).size() / num_threads.z + 1);
  pathline_block_kernel<<<num_blocks, num_threads>>>(
      dv, cuda::make_vec(cast<GpuReal>(g.min())),
      cuda::make_vec(cast<GpuReal>(g.max())),
      cuda::make_vec(cast<unsigned int>(g.resolution())), num_pathline_samples,
      pathline_buffer);
  return pathline_buffer;
}
//------------------------------------------------------------------------------
template <typename GpuReal = float, typename V, typename FieldReal,
          typename GridReal, typename SampleGridReal>
auto pathline_block(const field<V, FieldReal, 2, 2>& v, const grid<GridReal, 3>& g,
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
