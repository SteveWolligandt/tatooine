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
  static constexpr auto num_dimensions() {
    return N;
  }
  static constexpr auto vec_dim() {
    return VecDim;
  }
  static constexpr auto num_tex_channels() {
    return (VecDim == 3 ? 4 : VecDim);
  }
  static constexpr auto padding() {
    return num_tex_channels() - VecDim;
  }

  using vec_t = cuda::vec_t<Real, N>;
  using tex_t = tex<Real, num_tex_channels(), N>;

 private:
  tex_t m_sampler;
  vec_t m_min;
  vec_t m_max;

 private:
  template <typename Field, typename FieldReal, typename GridReal,
            typename TReal, size_t... Is,
            enable_if_arithmetic<TReal, GridReal> = true>
  steady_vectorfield(const field<Field, FieldReal, VecDim, VecDim>& f,
                     const grid<GridReal, N>& g, TReal t,
                     bool normalized_coords, std::index_sequence<Is...>)
      : m_sampler{sample_to_raw<Real>(f, g, t, padding()),
                  normalized_coords,
                  cuda::linear,
                  cuda::border,
                  g.dimension(Is).size()...},
        m_min{make_vec<Real>(g.dimension(Is).front()...)},
        m_max{make_vec<Real>(g.dimension(Is).back()...)} {}

 public:
  template <typename Field, typename FieldReal, typename GridReal,
            typename TReal, enable_if_arithmetic<TReal, GridReal> = true>
  steady_vectorfield(const field<Field, FieldReal, VecDim, VecDim>& f,
                     const grid<GridReal, N>& g, TReal t,
                     bool normalized_coords = true)
      : steady_vectorfield{f, g, t, normalized_coords,
                           std::make_index_sequence<N>{}} {}
  //----------------------------------------------------------------------------
  void free() { m_sampler.free(); }
  //----------------------------------------------------------------------------
  __device__ auto evaluate_uv(const cuda::vec_t<Real, N>& x) const {
    return m_sampler(x);
  }
  //----------------------------------------------------------------------------
  __device__ auto evaluate(const cuda::vec_t<Real, N>& x) const {
    return m_sampler(domain_pos_to_uv(x, min(), max(), resolution()));
  }
  //----------------------------------------------------------------------------
  __device__ auto operator()(const cuda::vec_t<Real, N>& x) const {
    return evaluate(x);
  }
  //----------------------------------------------------------------------------
  __host__ __device__ const auto& min() const { return m_min; }
  __host__ __device__ const auto& max() const { return m_max; }
  __host__ __device__ const auto& resolution() const {
    return m_sampler.resolution();
  }
};
template <typename Real, size_t N, size_t VecDim>
struct is_freeable<steady_vectorfield<Real, N, VecDim>> : std::true_type {};
template <typename Real, size_t N, size_t VecDim>
void free(steady_vectorfield<Real, N, VecDim>& f) {f.free();}

//=============================================================================
template <typename Real, size_t N, size_t VecDim>
class unsteady_vectorfield;

template <typename Real, size_t N, size_t VecDim>
struct is_freeable<unsteady_vectorfield<Real, N, VecDim>> : std::true_type {};

template <typename Real>
class unsteady_vectorfield<Real, 2, 2> {
 public:
  static constexpr auto num_dimensions() {
    return 2;
  }
  static constexpr auto vec_dim() {
    return 2;
  }
  static constexpr auto num_tex_channels() {
    return 2;
  }
  static constexpr auto padding() {
    return 0;
  }

  using vec_t = cuda::vec_t<Real, num_dimensions()>;
  using tex_t = tex<Real, num_tex_channels(), num_dimensions() + 1>;

 private:
  tex_t m_sampler;
  vec_t m_min;
  vec_t m_max;
  Real  m_tmin;
  Real  m_tmax;

 public:
  template <typename Field, typename FieldReal, typename GridReal,
            typename TReal, enable_if_arithmetic<TReal, GridReal> = true>
  unsteady_vectorfield(
      const field<Field, FieldReal, num_dimensions(), vec_dim()>& f,
      const grid<GridReal, num_dimensions()>& g, const linspace<TReal>& ts,
      bool normalized_coords)
      : m_sampler{sample_to_raw<Real>(f, g, ts, padding()),
                  normalized_coords,
                  cuda::linear,
                  cuda::border,
                  g.dimension(0).size(),
                  g.dimension(1).size(),
                  ts.size()},
        m_min{make_vec<Real>(g.dimension(0).front(), g.dimension(1).front())},
        m_max{make_vec<Real>(g.dimension(0).back(), g.dimension(1).back())},
        m_tmin(ts.front()),
        m_tmax(ts.back()) {}
  //----------------------------------------------------------------------------
  void free() { m_sampler.free(); }
  //----------------------------------------------------------------------------
  __device__ auto evaluate_uv(const cuda::vec_t<Real, num_dimensions() + 1>& uvw) const {
    return m_sampler(uvw);
  }
  //----------------------------------------------------------------------------
  __device__ auto evaluate(const cuda::vec_t<Real, num_dimensions()>& x,
                           Real t) const {
    return evaluate_uv(
        domain_pos_to_uv(x, min(), max(), t, tmin(), tmax(), resolution()));
  }
  //----------------------------------------------------------------------------
  __device__ auto operator()(const cuda::vec_t<Real, num_dimensions()>& x) const {
    return evaluate(x);
  }
  //----------------------------------------------------------------------------
  __host__ __device__ const auto& min() const { return m_min; }
  __host__ __device__ const auto& max() const { return m_max; }
  __host__ __device__ auto tmin() const { return m_tmin; }
  __host__ __device__ auto tmax() const { return m_tmax; }
  __host__ __device__ const auto& resolution() const {
    return m_sampler.resolution();
  }
};
template <typename Real, size_t N, size_t VecDim>
void free(unsteady_vectorfield<Real, N, VecDim>& f) {f.free();}
//==============================================================================
template <typename OutReal = float, typename Field, typename FieldReal, size_t N,
          typename GridReal, typename TReal,
          enable_if_arithmetic<TReal, GridReal> = true>
auto upload(const field<Field, FieldReal, N, N>& f,
            const grid<GridReal, N>& sample_grid,
            TReal t, bool normalized_coords = true) {
  return steady_vectorfield<OutReal, N, N>{f, sample_grid, t,
                                           normalized_coords};
}
//------------------------------------------------------------------------------
template <typename OutReal = float, typename Field, typename FieldReal, size_t N,
          typename GridReal, typename TReal,
          enable_if_arithmetic<TReal, GridReal> = true>
auto upload_normalized(const field<Field, FieldReal, N, N>& f,
                       const grid<GridReal, N>& sample_grid, TReal t,
                       bool normalized_coords = true) {
  return steady_vectorfield<OutReal, N, N>{normalize(f), sample_grid, t,
                                           normalized_coords};
}
//==============================================================================
template <typename OutReal = float, typename Field, typename FieldReal, size_t N,
          typename GridReal, typename TReal,
          enable_if_arithmetic<TReal, GridReal> = true>
auto upload(const field<Field, FieldReal, N, N>& f,
            const grid<GridReal, N>& sample_grid,
            const linspace<TReal>& ts, bool normalized_coords = true) {
  return unsteady_vectorfield<OutReal, N, N>{f, sample_grid, ts,
                                           normalized_coords};
}
//------------------------------------------------------------------------------
template <typename OutReal = float, typename Field, typename FieldReal, size_t N,
          typename GridReal, typename TReal,
          enable_if_arithmetic<TReal, GridReal> = true>
auto upload_normalized(const field<Field, FieldReal, N, N>& f,
                       const grid<GridReal, N>& sample_grid, const linspace<TReal> ts,
                       bool normalized_coords = true) {
  return unsteady_vectorfield<OutReal, N, N>{normalize(f), sample_grid, ts,
                                             normalized_coords};
}
//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
