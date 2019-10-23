#ifndef TATOOINE_CUDA_LIC_CUH
#define TATOOINE_CUDA_LIC_CUH

#include <tatooine/cuda/field_to_tex.cuh>
#include <tatooine/cuda/global_buffer.cuh>
#include <tatooine/cuda/lic.cuh>
#include <tatooine/cuda/math.cuh>
#include <tatooine/cuda/rungekutta4.cuh>
#include <tatooine/cuda/sample_field.cuh>
#include <tatooine/field.h>
#include <tatooine/linspace.h>
#include <tatooine/random.h>

#include <algorithm>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

__global__ void lic_kernel(cudaTextureObject_t v, cudaTextureObject_t noise_tex,
                             float* lic_tex, unsigned int num_samples,
                             float stepwidth, float2 min,
                             float2 max, uint2 res, uint2 noiseres) {
  const auto   globalIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);
  const size_t plainIdx  = globalIdx.x + globalIdx.y * res.x;
  if (globalIdx.x >= res.x || globalIdx.y >= res.y) { return; }

  const auto   pos     = global_idx_to_domain_pos2(globalIdx, min, max, res);
  auto         uv      = global_idx_to_uv2(globalIdx, res);
  const auto   fres           = make_float2(res.x, res.y);
  const auto   fnoiseres      = make_float2(noiseres.x, noiseres.y);
  auto         scale_noise_uv = fres / fnoiseres;
  auto         uv_noise       = uv * scale_noise_uv;
  float        lic_val = tex2D<float>(noise_tex, uv_noise.x, uv_noise.y);
  unsigned int cnt = 1;

  auto pos_integrated = pos;
  for (unsigned int i = 0; i < num_samples; ++i) {
    pos_integrated =
        rungekutta4_step_steady2(v, pos_integrated, stepwidth, min, max, res);

    if (isnan(pos_integrated.x) || isnan(pos_integrated.y)) { break; }
    uv       = domain_pos_to_uv2(pos_integrated, min, max, res);
    uv_noise = uv * scale_noise_uv;
    lic_val += tex2D<float>(noise_tex, uv_noise.x, uv_noise.y);
    ++cnt;
  }

  pos_integrated = pos;
  for (unsigned int i = 0; i < num_samples; ++i) {
    pos_integrated =
        rungekutta4_step_steady2(v, pos_integrated, -stepwidth, min, max, res);

    if (isnan(pos_integrated.x) || isnan(pos_integrated.y)) { break; }
    uv       = domain_pos_to_uv2(pos_integrated, min, max, res);
    uv_noise = uv * scale_noise_uv;
    lic_val += tex2D<float>(noise_tex, uv_noise.x, uv_noise.y);
    ++cnt;
  }

  //lic_tex[plainIdx] = clamp((lic_val / cnt) * 2.0f - 0.5f, 0.0f, 1.0f);
  lic_val = (lic_val / cnt) * 7.0f - 2.8f;
  if (lic_val < 0) {lic_val = 0;}
  if (lic_val > 1) {lic_val = 1;}
  lic_tex[plainIdx] = lic_val;
}

template <typename GPUReal = float, typename Field, typename FieldReal,
          typename XReal, typename YReal, typename TReal, typename SReal,
          typename RandEng>
auto call_lic_kernel(const field<Field, FieldReal, 2, 2>& v,
         const linspace<XReal>& x_domain, const linspace<YReal>& y_domain,
         TReal t, size_t num_samples, SReal stepwidth,
         RandEng&& rand_eng) {
  // sample and upload v to gpu
  const auto v_tex = normalized_to_tex<GPUReal>(v, x_domain, y_domain, t);

  // generate random noise texture
  //const size_t noise_tex_x = x_domain.size(), noise_tex_y = y_domain.size();
  const size_t noise_tex_x = 128, noise_tex_y = 128;
  std::vector<GPUReal>    noise(noise_tex_x * noise_tex_y);
  random_uniform<GPUReal, RandEng> rand{0.0f, 1.0f, rand_eng};
  std::generate(begin(noise), end(noise), [&rand] { return rand(); });
  texture_buffer<GPUReal, 1, 2> noise_tex(noise, true, linear, wrap,
                                          noise_tex_x, noise_tex_y);

  // lic tex memory
  global_buffer<GPUReal> lic_tex(x_domain.size() * y_domain.size());

  const dim3 num_threads{32, 32};
  const dim3 num_blocks(x_domain.size() / num_threads.x + 1,
                        y_domain.size() / num_threads.y + 1);
  lic_kernel<<<num_blocks, num_threads>>>(
      v_tex.device_ptr(), noise_tex.device_ptr(), lic_tex.device_ptr(),
      num_samples, stepwidth,
      make_float2(x_domain.front(), y_domain.front()),
      make_float2(x_domain.back(),  y_domain.back()),
      make_uint2 (x_domain.size(),  y_domain.size()),
      make_uint2 (noise_tex_x,  noise_tex_y));

  return lic_tex;
}

template <typename GPUReal = float, typename Field, typename FieldReal,
          typename XReal, typename YReal, typename TReal, typename SReal>
auto call_lic_kernel(const field<Field, FieldReal, 2, 2>& v,
                     const linspace<XReal>& x_domain,
                     const linspace<YReal>& y_domain, TReal t,
                     size_t num_samples, SReal stepwidth) {
  std::random_device dev;
  std::mt19937_64    randeng{dev()};
  return call_lic_kernel(v, x_domain, y_domain, t, num_samples, stepwidth,
                         randeng);
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
