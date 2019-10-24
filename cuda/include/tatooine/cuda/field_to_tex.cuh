#ifndef TATOOINE_CUDA_FIELD_TO_TEX_CUH
#define TATOOINE_CUDA_FIELD_TO_TEX_CUH

#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/linspace.h>
#include "coordinate_conversion.cuh"
#include "tex.cuh"

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T, size_t N>
class steady_vectorfield {
 public:
  using vec_t = cuda::vec<T, N>;

 private:
  tex<T, N, N>         m_sampler;
  var<cuda::vec<T, N>> m_min;
  var<cuda::vec<T, N>> m_max;

 public:
  template <typename Field, typename FieldReal, typename XReal, typename YReal,
            typename TReal, enable_if_arithmetic<TReal, XReal, YReal> = true>
  steady_vectorfield(const field<Field, FieldReal, 2, 2>& f,
                     const linspace<XReal>&               x_domain,
                     const linspace<YReal>& y_domain, TReal t,
                     bool normalized_coords = true)
      : m_sampler{sample_to_raw<T>(
                      f, grid<FieldReal, 2>{x_domain, y_domain}, t),
                  normalized_coords,
                  cuda::linear,
                  cuda::border,
                  x_domain.size(),
                  y_domain.size()},
        m_min{make_vec(static_cast<T>(x_domain.front()),
                       static_cast<T>(y_domain.front()))},
        m_max{make_vec(static_cast<T>(x_domain.back()),
                       static_cast<T>(y_domain.back()))} {}
  //----------------------------------------------------------------------------
  __device__ auto operator()(const cuda::vec<T, N>& x) const {
    return m_sampler(domain_pos_to_uv2(x, min(), max(), resolution()));
  }

  __device__ const auto& min() const { return *m_min; }
  __device__ const auto& max() const { return *m_max; }
  __device__ const auto& resolution() const { return m_sampler.resolution(); }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename RealOut = float, typename Field, typename FieldReal,
          typename TReal, typename XReal, typename YReal>
auto to_tex(const field<Field, FieldReal, 2, 2>& f,
            const linspace<XReal>& x_domain, const linspace<YReal>& y_domain,
            const linspace<TReal>& t_domain, bool normalized_coords = true) {
  auto resampled = sample_to_raw<RealOut>(
      f, grid<FieldReal, 2>{x_domain, y_domain}, t_domain);
  return cuda::tex<RealOut, 2, 3>{
      resampled,       normalized_coords, cuda::linear,   cuda::border,
      x_domain.size(), y_domain.size(),   t_domain.size()};
}
//------------------------------------------------------------------------------
template <typename RealOut = float, typename Field, typename FieldReal,
          typename TReal, typename XReal, typename YReal>
auto normalized_to_tex(const field<Field, FieldReal, 2, 2>& f,
                       const linspace<XReal>&               x_domain,
                       const linspace<YReal>& y_domain, TReal t,
                       bool normalized_coords = true) {
  auto resampled = sample_normalized_to_raw<RealOut>(
      f, grid<FieldReal, 2>{x_domain, y_domain}, t);
  return cuda::tex<RealOut, 2, 2>{resampled,       normalized_coords,
                                             cuda::linear,    cuda::border,
                                             x_domain.size(), y_domain.size()};
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
