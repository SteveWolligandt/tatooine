#ifndef TATOOINE_CUDA_LIC_CUH
#define TATOOINE_CUDA_LIC_CUH

#include <tatooine/field.h>
#include <tatooine/linspace.h>
#include <tatooine/random.h>

#include <algorithm>
#include <tatooine/cuda/buffer.cuh>
#include <tatooine/cuda/field.cuh>
#include <tatooine/cuda/lic.cuh>
#include <tatooine/cuda/math.cuh>
#include <tatooine/cuda/rungekutta4.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename FieldReal, typename NoiseReal, typename StepwidthReal,
          typename OutReal                    = NoiseReal,
          enable_if_arithmetic<StepwidthReal> = true>
__global__ void lic_kernel(steady_vectorfield<FieldReal, 2, 2> v,
                           tex<NoiseReal, 1, 2>             noise_tex,
                           buffer<OutReal> lic_tex, unsigned int num_samples,
                           StepwidthReal stepwidth, float2 min, float2 max, uint2 res) {
  const auto globalIdx =
      make_vec_promoted(blockIdx.x * blockDim.x + threadIdx.x,
                        blockIdx.y * blockDim.y + threadIdx.y);
  if (globalIdx.x >= res.x || globalIdx.y >= res.y) { return; }
  const auto   noiseres  = noise_tex.resolution();
  const auto   fres      = make_vec<float>(res.x, res.y);
  const auto   fnoiseres = make_vec<float>(noiseres.x, noiseres.y);
  const size_t plainIdx  = globalIdx.x + globalIdx.y * res.x;

  const auto pos = global_idx_to_domain_pos(globalIdx, min, max, res);
  auto       uv  = global_idx_to_uv(globalIdx, res);
  auto       scale_noise_uv = fres / fnoiseres;
  auto       uv_noise       = uv * scale_noise_uv;
  auto       lic_val        = noise_tex(uv_noise);
  unsigned int cnt          = 1;

  // forward integration
  auto pos_integrated = pos;
  for (unsigned int i = 0; i < num_samples; ++i) {
    pos_integrated = rungekutta4_step(v, pos_integrated, stepwidth);

    if (isnan(pos_integrated.x) || isnan(pos_integrated.y)) { break; }
    uv       = domain_pos_to_uv(pos_integrated, v.min(), v.max(), res);
    uv_noise = uv * scale_noise_uv;
    lic_val += noise_tex(uv_noise);
    ++cnt;
  }

  // forward integration
  pos_integrated = pos;
  for (unsigned int i = 0; i < num_samples; ++i) {
    pos_integrated = rungekutta4_step(v, pos_integrated, -stepwidth);

    if (isnan(pos_integrated.x) || isnan(pos_integrated.y)) { break; }
    uv       = domain_pos_to_uv(pos_integrated, v.min(), v.max(), res);
    uv_noise = uv * scale_noise_uv;
    lic_val += noise_tex(uv_noise);
    ++cnt;
  }

  // contrast stretch
  lic_val = (lic_val / cnt) * 11.0f - 4.7f;
  if (lic_val < 0) { lic_val = 0; }
  if (lic_val > 1) { lic_val = 1; }
  lic_tex[plainIdx] = lic_val;
}

template <typename GPUReal = float, typename Field, typename FieldReal,
          typename SampleGridReal, typename PixelGridReal, typename TReal, typename SReal, typename RandEng>
auto call_lic_kernel(const field<Field, FieldReal, 2, 2>& v,
                     const grid<SampleGridReal, 2>&       vf_sample_grid,
                     const grid<PixelGridReal, 2>& pixel_grid, TReal t,
                     size_t num_samples, SReal stepwidth, RandEng&& rand_eng) {
  // sample and upload v to gpu
  auto cuV = upload_normalized<GPUReal>(v, vf_sample_grid, t);

  // generate random noise texture
  // const size_t noise_tex_x = x_domain.size(), noise_tex_y = y_domain.size();
  const size_t                     noise_tex_x = 128, noise_tex_y = 128;
  std::vector<GPUReal>             noise(noise_tex_x * noise_tex_y);
  random_uniform<GPUReal, RandEng> rand{0, 1, rand_eng};
  std::generate(begin(noise), end(noise), [&rand] { return rand(); });
  tex<GPUReal, 1, 2> noise_tex(noise, true, linear, wrap, noise_tex_x,
                               noise_tex_y);

  // lic tex memory
  buffer<GPUReal> lic_tex(pixel_grid.num_vertices());

  const dim3 num_threads{32, 32};
  const dim3 num_blocks(pixel_grid.dimension(0).size() / num_threads.x + 1,
                        pixel_grid.dimension(1).size() / num_threads.y + 1);
  lic_kernel<<<num_blocks, num_threads>>>(
      cuV, noise_tex, lic_tex, num_samples, static_cast<GPUReal>(stepwidth),
      make_float2(pixel_grid.dimension(0).front(),
                  pixel_grid.dimension(1).front()),
      make_float2(pixel_grid.dimension(0).back(),
                  pixel_grid.dimension(1).back()),
      make_uint2(pixel_grid.dimension(0).size(),
                 pixel_grid.dimension(1).size()));
  
  free(noise_tex, cuV);
  return lic_tex;
}

template <typename GPUReal = float, typename Field, typename FieldReal,
          typename SampleGridReal, typename PixelGridReal, typename TReal,
          typename SReal>
auto call_lic_kernel(const field<Field, FieldReal, 2, 2>& v,
                     const grid<SampleGridReal, 2>&       vf_sample_grid,
                     const grid<PixelGridReal, 2>& pixel_grid, TReal t,
                     size_t num_samples, SReal stepwidth) {
  std::random_device dev;
  std::mt19937_64    randeng{dev()};
  return call_lic_kernel(v, vf_sample_grid, pixel_grid, t, num_samples,
                         stepwidth, randeng);
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
