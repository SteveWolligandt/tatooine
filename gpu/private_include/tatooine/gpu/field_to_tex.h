#ifndef TATOOINE_GPU_FIELD_TO_TEX_H
#define TATOOINE_GPU_FIELD_TO_TEX_H

#include <tatooine/field.h>
//#include <tatooine/grid_sampler.h>
#include <tatooine/cuda/texture_buffer.h>

//==============================================================================
namespace tatooine{
namespace gpu{
//==============================================================================

template <typename Field, typename FieldReal, typename TReal>
auto to_tex(const field<Field, FieldReal, 2>& f, TReal t,
            const linspace<Real>& x_domain, const linspace<Real>& y_domain) {
  auto resampled = resample(f, grid<FieldReal, 2>{x_domain, y_domain});
  return texture_buffer<float, 2, 2>{resampled.sampler().data().unchunk(),
                                     x_domain.size(), y_domain.size()};
}

//==============================================================================
}
}
//==============================================================================

#endif
