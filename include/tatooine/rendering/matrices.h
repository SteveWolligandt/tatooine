#ifndef TATOOINE_RENDERING_MATRICES_H
#define TATOOINE_RENDERING_MATRICES_H
//==============================================================================
#include <tatooine/mat.h>
#include <cmath>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename Real>
constexpr auto translation_matrix(Real x, Real y, Real z) {
  auto m  = mat<Real, 4, 4>::eye();
  m(0, 3) = x;
  m(1, 3) = y;
  m(2, 3) = z;
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto translation_matrix(vec<Real, 3> t) {
  auto m  = mat<Real, 4, 4>::eye();
  m(0, 3) = t(0);
  m(1, 3) = t(1);
  m(2, 3) = t(2);
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto scale_matrix(Real s) {
  auto m  = mat<Real, 4, 4>::zeros();
  m(0, 0) = s;
  m(1, 1) = s;
  m(2, 2) = s;
  m(3, 3) = 1;
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto scale_matrix(Real x, Real y, Real z) {
  auto m  = mat<Real, 4, 4>::zeros();
  m(0, 0) = x;
  m(1, 1) = y;
  m(2, 2) = z;
  m(3, 3) = 1;
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto scale_matrix(const vec<Real, 3>& s) {
  auto m  = mat<Real, 4, 4>::zeros();
  m(0, 0) = s(0);
  m(1, 1) = s(1);
  m(2, 2) = s(2);
  m(3, 3) = 1;
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto rotation_matrix(Real angle, Real u, Real v, Real w)
    -> mat<Real, 4, 4> {
  const Real s = std::sin(angle);
  const Real c = std::cos(angle);
  return {{u * u + (v * v + w * w) * c, u * v * (1 - c) - w * s,
           u * w * (1 - c) + v * s, Real(0)},
          {u * v + (1 - c) + w * s, v * v * (u * u + w * w) * c,
           v * w * (1 - c) + u * s, Real(0)},
          {u * w + (1 - c) - v * s, v * v * (1 - c) + u * s,
           w * w + (u * u + v * v) * c, Real(0)},
          {Real(0), Real(0), Real(0), Real(1)}};
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto rotation_matrix(Real angle, const vec<Real, 3>& axis) {
  return rotation_matrix(angle, axis(0), axis(1), axis(2));
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto orthographic_matrix(const Real l, const Real r, const Real b,
                                   const Real t, const Real n, const Real f)
    -> mat<Real, 4, 4> {
  return {{2 / (r - l), Real(0), Real(0), -(r + l) / (r - l)},
          {Real(0), 2 / (t - b), Real(0), -(t + b) / (t - b)},
          {Real(0), Real(0), -2 / (f - n), -(f + n) / (f - n)},
          {Real(0), Real(0), Real(0), Real(1)}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto look_at_matrix(const vec<Real, 3>& eye, const vec<Real, 3>& center,
                    const vec<Real, 3>& up = {0, 1, 0}) -> mat<Real, 4, 4> {
  const auto D = normalize(eye - center);
  const auto R = normalize(cross(up, D));
  const auto U = cross(D, R);
  return {{R(0), U(0), D(0), eye(0)},
          {R(1), U(1), D(1), eye(1)},
          {R(2), U(2), D(2), eye(2)},
          {Real(0), Real(0), Real(0), Real(1)}};
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
