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
  auto m  = Mat4<Real>::eye();
  m(0, 3) = x;
  m(1, 3) = y;
  m(2, 3) = z;
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto translation_matrix(Vec3<Real> t) {
  auto m  = Mat4<Real>::eye();
  m(0, 3) = t(0);
  m(1, 3) = t(1);
  m(2, 3) = t(2);
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto scale_matrix(Real s) {
  auto m  = Mat4<Real>::zeros();
  m(0, 0) = s;
  m(1, 1) = s;
  m(2, 2) = s;
  m(3, 3) = 1;
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto scale_matrix(Real x, Real y, Real z) {
  auto m  = Mat4<Real>::zeros();
  m(0, 0) = x;
  m(1, 1) = y;
  m(2, 2) = z;
  m(3, 3) = 1;
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto scale_matrix(Vec3<Real> const& s) {
  auto m  = Mat4<Real>::zeros();
  m(0, 0) = s(0);
  m(1, 1) = s(1);
  m(2, 2) = s(2);
  m(3, 3) = 1;
  return m;
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto rotation_matrix(Real angle, Real u, Real v, Real w)
    -> Mat4<Real> {
  Real const s = std::sin(angle);
  Real const c = std::cos(angle);
  return Mat4<Real>{{u * u + (v * v + w * w) * c, u * v * (1 - c) - w * s,
           u * w * (1 - c) + v * s, Real(0)},
          {u * v + (1 - c) + w * s, v * v * (u * u + w * w) * c,
           v * w * (1 - c) + u * s, Real(0)},
          {u * w + (1 - c) - v * s, v * v * (1 - c) + u * s,
           w * w + (u * u + v * v) * c, Real(0)},
          {Real(0), Real(0), Real(0), Real(1)}};
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto rotation_matrix(Real angle, Vec3<Real> const& axis) {
  return rotation_matrix(angle, axis(0), axis(1), axis(2));
}
//------------------------------------------------------------------------------
template <typename Real>
constexpr auto orthographic_matrix(Real const l, Real const r, Real const b,
                                   Real const t, Real const n, Real const f)
    -> Mat4<Real> {
  return Mat4<Real>{{2 / (r - l), Real(0), Real(0), -(r + l) / (r - l)},
          {Real(0), 2 / (t - b), Real(0), -(t + b) / (t - b)},
          {Real(0), Real(0), -2 / (f - n), -(f + n) / (f - n)},
          {Real(0), Real(0), Real(0), Real(1)}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto look_at_matrix(Vec3<Real> const& eye, Vec3<Real> const& center,
                    Vec3<Real> const& up = {0, 1, 0}) -> Mat4<Real> {
  static auto constexpr O = Real(0);
  static auto constexpr I = Real(1);

  auto const zaxis = normalize(eye - center);
  auto const xaxis = normalize(cross(up, zaxis));
  auto const yaxis = cross(zaxis, xaxis);
  return Mat4<Real>{{xaxis.x(), xaxis.y(), xaxis.z(), -dot(xaxis, eye)},
                    {yaxis.x(), yaxis.y(), yaxis.z(), -dot(yaxis, eye)},
                    {zaxis.x(), zaxis.y(), zaxis.z(), -dot(zaxis, eye)},
                    {   O,    O,    O,           I}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr frustum_matrix(Real const l, Real const r,
                       Real const b, Real const t,
                       Real const n, Real const f) -> Mat4<Real> {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);

  auto const iw  = 1 / (r - l);
  auto const xs  = 2 * n * iw;
  auto const xzs = (r + l) * iw;

  auto const ih  = 1 / (t - b);
  auto const ys  = 2 * n * ih;
  auto const yzs = (t + b) * ih;

  auto const ind   = 1 / (f - n);
  auto const zs    = -(f + n) * ind;
  auto const zt    = -2 * f * n * ind;

  return Mat4<Real>{{xs,  O, xzs,  O},
                    { O, ys, yzs,  O},
                    { O,  O,  zs, zt},
                    { O,  O,  -I,  O}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr perspective_matrix(Real const angle_of_view, Real const aspect_ratio,
                           Real const n, Real const f) {
  auto constexpr angle_scale = Real(1) / Real(2) * Real(M_PI) / Real(180);
  auto const scale           = gcem::tan(angle_of_view * angle_scale) * n;
  auto const r               = aspect_ratio * scale;
  auto const l               = -r;
  auto const t               = scale;
  auto const b               = -t;
  return frustum_matrix(l, r, b, t, n, f);
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
