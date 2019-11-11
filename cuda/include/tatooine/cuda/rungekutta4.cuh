#ifndef TATOOINE_CUDA_RUNGEKUTTA4_CUH
#define TATOOINE_CUDA_RUNGEKUTTA4_CUH

#include <tatooine/cuda/math.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename Real>
__device__ vec_t<Real, 2> rungekutta4_step(
    const steady_vectorfield<Real, 2, 2>& v, const vec_t<Real, 2>& pos,
    Real stepwidth) {
  const auto k1 = stepwidth * v(pos);

  const auto x2 = pos + k1 / 2;
  if (x2.x < v.min().x || x2.x > v.max().x ||
      x2.y < v.min().y || x2.y > v.max().y) {
    return make_vec<Real>(0.0 / 0.0, 0.0 / 0.0);
  }
  const auto k2 = stepwidth * v(x2);

  const auto x3 = pos + k2 / 2;
  if (x3.x < v.min().x || x3.x > v.max().x ||
      x3.y < v.min().y || x3.y > v.max().y) {
    return make_vec<Real>(0.0 / 0.0, 0.0 / 0.0);
  }
  const auto k3 = stepwidth * v(x3);

  const auto x4 = pos + k3;
  if (x4.x < v.min().x || x4.x > v.max().x || x4.y < v.min().y ||
      x4.y > v.max().y) {
    return make_vec<Real>(0.0 / 0.0, 0.0 / 0.0);
  }
  const auto k4 = stepwidth * v(x4);

  const auto stepped = pos + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
  if (stepped.x < v.min().x || stepped.x > v.max().x ||
      stepped.y < v.min().y || stepped.y > v.max().y) {
    return make_vec<Real>(0.0 / 0.0, 0.0 / 0.0);
  }
  return stepped;
}

//------------------------------------------------------------------------------
template <typename Real>
__device__ vec_t<Real, 2> rungekutta4_step(
    const unsteady_vectorfield<Real, 2, 2>& v, const vec_t<Real, 2>& pos,
    Real t, Real stepwidth) {
  if (pos.x < v.min().x || pos.x > v.max().x ||
      pos.y < v.min().y || pos.y > v.max().y ||
      t < v.tmin() || t > v.tmax() ||
      t + stepwidth < v.tmin() || t + stepwidth > v.tmax()) {
    return make_vec<Real>(0.0 / 0.0, 0.0 / 0.0);
  }

  const auto k1 = stepwidth * v(pos, t);

  const auto x2 = pos + k1 / 2;
  if (x2.x < v.min().x || x2.x > v.max().x ||
      x2.y < v.min().y || x2.y > v.max().y) {
    return make_vec<Real>(0.0 / 0.0, 0.0 / 0.0);
  }
  const auto k2 = stepwidth * v(x2, t + stepwidth / 2);

  const auto x3 = pos + k2 / 2;
  if (x3.x < v.min().x || x3.x > v.max().x ||
      x3.y < v.min().y || x3.y > v.max().y) {
    return make_vec<Real>(0.0 / 0.0, 0.0 / 0.0);
  }
  const auto k3 = stepwidth * v(x3, t + stepwidth / 2);

  const auto x4 = pos + k3;
  if (x4.x < v.min().x || x4.x > v.max().x ||
      x4.y < v.min().y || x4.y > v.max().y) {
    return make_vec<Real>(0.0 / 0.0, 0.0 / 0.0);
  }
  const auto k4 = stepwidth * v(x4, t + stepwidth);

  const auto stepped = pos + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
  if (stepped.x < v.min().x || stepped.x > v.max().x ||
      stepped.y < v.min().y || stepped.y > v.max().y) {
    return make_vec<Real>(0.0 / 0.0, 0.0 / 0.0);
  }
  return stepped;
}

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
