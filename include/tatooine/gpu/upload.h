#ifndef TATOOINE_GPU_FIELD_TO_GPU_H
#define TATOOINE_GPU_FIELD_TO_GPU_H
//==============================================================================
#include <yavin/texture.h>
#include <tatooine/field.h>
#include <tatooine/grid_sampler.h>
//==============================================================================
namespace tatooine::gpu {
//==============================================================================
template <typename GPUReal = float, typename Real,
          template <typename> typename InterpolatorX,
          template <typename> typename InterpolatorY>
auto upload(const sampled_field<
            grid_sampler<Real, 2, vec<Real, 2>, InterpolatorX, InterpolatorY>,
            Real, 2, 2>& v) {
  using namespace yavin;
  const std::vector<Real> data = v.sampler().data().unchunk_plain();
  return texture<2, GPUReal, RG>(data, v.sampler().size(0), v.sampler().size(1));
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename V, typename Real>
auto upload(const field<V, Real, 2, 2>& v, const grid<Real, 2>& discrete_domain,
            Real t) {
  using namespace interpolation;
  return upload<GPUReal>(resample<linear, linear>(v, discrete_domain, t));
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename V, typename Real>
auto upload(const field<V, Real, 2, 2>& v,
            const linspace<Real>& xres,
            const linspace<Real>& yres, Real t) {
  return upload<GPUReal>(v, grid{xres, yres}, t);
}
//==============================================================================
template <typename GPUReal = float, typename Real,
          template <typename> typename InterpolatorX,
          template <typename> typename InterpolatorY,
          template <typename> typename InterpolatorZ>
auto upload(
    const sampled_field<grid_sampler<Real, 3, vec<Real, 3>, InterpolatorX,
                                     InterpolatorY, InterpolatorZ>,
                        Real, 2, 2>& v) {
  using namespace yavin;
  const auto data = v.sampler().data().unchunk_plain();
  return texture<3, GPUReal, RG>(v.sampler().size(0),
                                 v.sampler().size(1),
                                 v.sampler().size(2));
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename V, typename Real>
auto upload(const field<V, Real, 2, 2>& v, const grid<Real, 2>& discrete_domain,
            linspace<Real> tres) {
  using namespace interpolation;
  return upload<GPUReal>(
      resample<linear, linear, linear>(v, discrete_domain, tres));
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename V, typename Real>
auto upload(const field<V, Real, 2, 2>& v,
            const linspace<Real>& xres,
            const linspace<Real>& yres, linspace<Real> tres) {
  return upload<GPUReal>(v, grid{xres, yres}, tres);
}
//==============================================================================
template <typename GPUReal = float, typename Real,
          template <typename> typename InterpolatorX,
          template <typename> typename InterpolatorY>
auto upload(const sampled_field<grid_sampler<Real, 3, vec<Real, 3>,
                                             InterpolatorX, InterpolatorY>,
                                Real, 3, 3>& v) {
  using namespace yavin;
  const auto data = v.sampler().data().unchunk_plain();
  return texture<3, GPUReal, RGB>(v.sampler().size(0), v.sampler().size(1),
                                  v.sampler().size(2));
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename V, typename Real>
auto upload(const field<V, Real, 3, 3>& v, const grid<Real, 3>& discrete_domain,
            Real t) {
  using namespace interpolation;
  return upload<GPUReal>(
      resample<linear, linear, linear>(v, discrete_domain, t));
}
//------------------------------------------------------------------------------
template <typename GPUReal = float, typename V, typename Real>
auto upload(const field<V, Real, 3, 3>& v,
            const linspace<Real>& xres,
            const linspace<Real>& yres,
            const linspace<Real>& zres, Real t) {
  return upload<GPUReal>(v, grid{xres, yres, zres}, t);
}
//==============================================================================
}  // namespace tatooine::gpu
//==============================================================================

#endif
