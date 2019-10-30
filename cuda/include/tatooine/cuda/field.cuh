#ifndef TATOOINE_CUDA_FIELD_CUH
#define TATOOINE_CUDA_FIELD_CUH

#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/cuda/type_traits.cuh>
#include <tatooine/cuda/coordinate_conversion.cuh>
#include <tatooine/cuda/tex.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename Real, size_t N, size_t VecDim>
class steady_vectorfield {
 public:
  using vec_t = cuda::vec_t<Real, N>;

 private:
  tex<Real, N, N>         m_sampler;
  var<cuda::vec_t<Real, N>> m_min;
  var<cuda::vec_t<Real, N>> m_max;

 public:
  template <typename Field, typename FieldReal, typename GridReal,
            typename TReal, enable_if_arithmetic<TReal, GridReal> = true>
  steady_vectorfield(const field<Field, FieldReal, VecDim, VecDim>& f,
                     const grid<GridReal, N>& g, TReal t,
                     bool normalized_coords = true)
      : m_sampler{sample_to_raw<Real>(f, g, t),
                  normalized_coords,
                  cuda::linear,
                  cuda::border,
                  g.dimension(0).size(),
                  g.dimension(1).size()},
        m_min{make_vec<Real>(g.dimension(0).front(), g.dimension(1).front())},
        m_max{make_vec<Real>(g.dimension(0).back(), g.dimension(1).back())} {}
  //----------------------------------------------------------------------------
  __device__ auto evaluate(const cuda::vec_t<Real, N>& x) const {
    return m_sampler(domain_pos_to_uv(x, min(), max(), resolution()));
  }
  //----------------------------------------------------------------------------
  __device__ auto operator()(const cuda::vec_t<Real, N>& x) const {
    return evaluate(x);
  }
  //----------------------------------------------------------------------------
  __device__ const auto& min() const { return *m_min; }
  __device__ const auto& max() const { return *m_max; }
  __device__ const auto& resolution() const { return m_sampler.resolution(); }
};
//==============================================================================
template <typename OutReal = float, typename Field, typename FieldReal,
          typename GridReal, typename TReal,
          enable_if_arithmetic<TReal, GridReal> = true>
auto upload(const field<Field, FieldReal, 2, 2>& f,
            const grid<GridReal, 2>& sample_grid,
            TReal t, bool normalized_coords = true) {
  return steady_vectorfield<OutReal, 2, 2>{f, sample_grid, t,
                                           normalized_coords};
}
//------------------------------------------------------------------------------
template <typename OutReal = float, typename Field, typename FieldReal,
          typename GridReal, typename TReal,
          enable_if_arithmetic<TReal, GridReal> = true>
auto upload_normalized(const field<Field, FieldReal, 2, 2>& f,
                       const grid<GridReal, 2>& sample_grid, TReal t,
                       bool normalized_coords = true) {
  return steady_vectorfield<OutReal, 2, 2>{normalize(f), sample_grid, t,
                                           normalized_coords};
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
