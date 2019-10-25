#ifndef TATOOINE_CUDA_RUNGEKUTTA4_CUH
#define TATOOINE_CUDA_RUNGEKUTTA4_CUH

#include <tatooine/cuda/math.cuh>
#include <tatooine/cuda/sample_field.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename Real, size_t N>
__device__ vec_t<Real, N> rungekutta4_step(
    const steady_vectorfield<Real, N, N>& v, const vec_t<Real, N>& pos,
    Real stepwidth) {
  const auto k1 = stepwidth * v(pos);

  const auto x2 = pos + k1 * 0.5f;
  if (x2.x < v.min().x || x2.y > v.max().x || x2.y < v.min().y ||
      x2.y > v.max().y) {
    return make_vec<Real>(0.0f / 0.0f, 0.0f / 0.0f);
  }
  const auto k2 = stepwidth * v(x2);

  const auto x3 = pos + k2 * 0.5f;
  if (x3.x < v.min().x || x3.y > v.max().x || x3.y < v.min().y ||
      x3.y > v.max().y) {
    return make_vec<Real>(0.0f / 0.0f, 0.0f / 0.0f);
  }
  const auto k3 = stepwidth * v(x3);

  const auto x4 = pos + k3;
  if (x4.x < v.min().x || x4.y > v.max().x || x4.y < v.min().y ||
      x4.y > v.max().y) {
    return make_vec<Real>(0.0f / 0.0f, 0.0f / 0.0f);
  }
  const auto k4      = stepwidth * v(x4);

  const auto stepped = pos + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0f;
  if (stepped.x < v.min().x || stepped.y > v.max().x || stepped.y < v.min().y ||
      stepped.y > v.max().y) {
    return make_vec<Real>(0.0f / 0.0f, 0.0f / 0.0f);
  }
  return stepped;
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
