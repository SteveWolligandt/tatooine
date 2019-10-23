#ifndef TATOOINE_CUDA_FIELD_TO_TEX_CUH
#define TATOOINE_CUDA_FIELD_TO_TEX_CUH

#include <tatooine/cuda/texture_buffer.cuh>
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/linspace.h>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename RealOut = float, typename Field, typename FieldReal,
          typename TReal, typename XReal, typename YReal,
          enable_if_arithmetic<TReal, XReal, YReal> = true>
auto to_tex(const field<Field, FieldReal, 2, 2>& f,
            const linspace<XReal>& x_domain, const linspace<YReal>& y_domain,
            TReal t, bool normalized_coords = true) {
  auto resampled =
      sample_to_raw<RealOut>(f, grid<FieldReal, 2>{x_domain, y_domain}, t);
  return cuda::texture_buffer<RealOut, 2, 2>{resampled,       normalized_coords,
                                             cuda::linear,    cuda::border,
                                             x_domain.size(), y_domain.size()};
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename RealOut = float, typename Field, typename FieldReal,
          typename TReal, typename XReal, typename YReal>
auto to_tex(const field<Field, FieldReal, 2, 2>& f,
            const linspace<XReal>& x_domain, const linspace<YReal>& y_domain,
            const linspace<TReal>& t_domain, bool normalized_coords = true) {
  auto resampled =
      sample_to_raw<RealOut>(f, grid<FieldReal, 2>{x_domain, y_domain}, t_domain);
  return cuda::texture_buffer<RealOut, 2, 3>{
      resampled,       normalized_coords, cuda::linear,   cuda::border,
      x_domain.size(), y_domain.size(),   t_domain.size()};
}
//------------------------------------------------------------------------------
template <typename RealOut = float, typename Field, typename FieldReal,
          typename TReal, typename XReal, typename YReal>
auto normalized_to_tex(const field<Field, FieldReal, 2, 2>& f,
                       const linspace<XReal>&            x_domain,
                       const linspace<YReal>& y_domain, TReal t,
                       bool normalized_coords = true) {
  auto resampled = sample_normalized_to_raw<RealOut>(
      f, grid<FieldReal, 2>{x_domain, y_domain}, t);
  return cuda::texture_buffer<RealOut, 2, 2>{resampled,       normalized_coords,
                                             cuda::linear,    cuda::border,
                                             x_domain.size(), y_domain.size()};
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
