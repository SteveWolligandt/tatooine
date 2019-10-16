#ifndef TATOOINE_GPU_FIELD_TO_TEX_H
#define TATOOINE_GPU_FIELD_TO_TEX_H

#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/linspace.h>
#include <tatooine/cuda/texture_buffer.h>

//==============================================================================
namespace tatooine {
namespace gpu {
//==============================================================================

template <typename RealOut = float, typename Field, typename FieldReal,
          typename TReal, typename LinRealX, typename LinRealY>
auto to_tex(const field<Field, FieldReal, 2, 2>& f,
            const linspace<LinRealX>&            x_domain,
            const linspace<LinRealY>& y_domain, TReal t) {
  auto resampled =
      sample_to_raw<RealOut>(f, grid<FieldReal, 2>{x_domain, y_domain}, t);
  return cuda::texture_buffer<RealOut, 2, 2>{resampled, x_domain.size(),
                                           y_domain.size()};
}

//==============================================================================
}  // namespace gpu
}  // namespace tatooine
//==============================================================================

#endif
