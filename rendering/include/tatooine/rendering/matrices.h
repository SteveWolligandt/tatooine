#ifndef TATOOINE_RENDERING_MATRICES_H
#define TATOOINE_RENDERING_MATRICES_H
//==============================================================================
#include <tatooine/mat.h>

#include <cmath>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename Real>
auto constexpr translation_matrix(Real const x, Real const y, Real const z) {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);
  return Mat4<Real>{{I, O, O, x}, {O, I, O, y}, {O, O, I, z}, {O, O, O, I}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr translation_matrix(Vec3<Real> const& t) {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);
  return Mat4<Real>{
      {I, O, O, t.x()}, {O, I, O, t.y()}, {O, O, I, t.z()}, {O, O, O, I}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr scale_matrix(Real s) {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);
  return Mat4<Real>{{s, O, O, O}, {O, s, O, O}, {O, O, s, O}, {O, O, O, I}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr scale_matrix(Real x, Real y, Real z) {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);
  return Mat4<Real>{{x, O, O, O}, {O, y, O, O}, {O, O, z, O}, {O, O, O, I}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr scale_matrix(Vec3<Real> const& s) {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);
  return Mat4<Real>{
      {s.x(), O, O, O}, {O, s.y(), O, O}, {O, O, s.z(), O}, {O, O, O, I}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr rotation_matrix(Real angle, Real u, Real v, Real w)
    -> Mat4<Real> {
  Real const s     = gcem::sin(angle);
  Real const c     = gcem::cos(angle);
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);
  return Mat4<Real>{{u * u + (v * v + w * w) * c, u * v * (1 - c) - w * s,
                     u * w * (1 - c) + v * s, O},
                    {u * v + (1 - c) + w * s, v * v * (u * u + w * w) * c,
                     v * w * (1 - c) + u * s, O},
                    {u * w + (1 - c) - v * s, v * v * (1 - c) + u * s,
                     w * w + (u * u + v * v) * c, O},
                    {O, O, O, I}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr rotation_matrix(Real angle, Vec3<Real> const& axis) {
  return rotation_matrix(angle, axis(0), axis(1), axis(2));
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr orthographic_matrix(Real const left, Real const right,
                                   Real const bottom, Real const top,
                                   Real const near, Real const far) {
  auto constexpr O      = Real(0);
  auto constexpr I      = Real(1);
  auto const width      = right - left;
  auto const inv_width  = 1 / width;
  auto const height     = top - bottom;
  auto const inv_height = 1 / height;
  auto const depth      = far - near;
  auto const inv_depth  = 1 / depth;

  return Mat4<Real>{{2 * inv_width, O, O, -(right + left) * inv_width},
                    {O, 2 * inv_height, O, -(top + bottom) * inv_height},
                    {O, O, -2 * inv_depth, -(far + near) * inv_depth},
                    {O, O, O, I}};
}
//==============================================================================
/// Can be used as transform matrix of an object.
template <typename Real>
auto constexpr look_at_matrix_left_hand(Vec3<Real> const& eye,
                                        Vec3<Real> const& center,
                                        Vec3<Real> const& up = {0, 1, 0})
    -> Mat4<Real> {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);

  auto const zaxis = normalize(center - eye);
  auto const xaxis = cross(zaxis, normalize(up));
  auto const yaxis = cross(xaxis, zaxis);
  return Mat4<Real>{{xaxis.x(), yaxis.x(), zaxis.x(), eye.x()},
                    {xaxis.y(), yaxis.y(), zaxis.y(), eye.y()},
                    {xaxis.z(), yaxis.z(), zaxis.z(), eye.z()},
                    {O, O, O, I}};
}
//------------------------------------------------------------------------------
/// Can be used as view matrix of a camera.
template <typename Real>
auto constexpr inv_look_at_matrix_left_hand(Vec3<Real> const& eye,
                                            Vec3<Real> const& center,
                                            Vec3<Real> const& up = {0, 1, 0}) {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);

  auto const zaxis = normalize(center - eye);
  auto const xaxis = cross(zaxis, normalize(up));
  auto const yaxis = cross(xaxis, zaxis);

  return Mat4<Real>{{xaxis.x(), xaxis.y(), xaxis.z(), -dot(xaxis, eye)},
                    {yaxis.x(), yaxis.y(), yaxis.z(), -dot(yaxis, eye)},
                    {zaxis.x(), zaxis.y(), zaxis.z(), -dot(zaxis, eye)},
                    {O, O, O, I}};
}
//------------------------------------------------------------------------------
/// Can be used as transform matrix of an object.
template <typename Real>
auto constexpr look_at_matrix_right_hand(Vec3<Real> const& eye,
                                         Vec3<Real> const& center,
                                         Vec3<Real> const& up = {0, 1, 0}) {
  auto const zaxis = normalize(eye - center);
  auto const xaxis = cross(normalize(up), zaxis);
  auto const yaxis = cross(zaxis, xaxis);
  return Mat4<Real>{{xaxis.x(), yaxis.x(), zaxis.x(), eye.x()},
                    {xaxis.y(), yaxis.y(), zaxis.y(), eye.y()},
                    {xaxis.z(), yaxis.z(), zaxis.z(), eye.z()},
                    {Real(0), Real(0), Real(0), Real(1)}};
}
//------------------------------------------------------------------------------
/// Can be used as view matrix of a camera.
template <typename Real>
auto constexpr inv_look_at_matrix_right_hand(Vec3<Real> const& eye,
                                             Vec3<Real> const& center,
                                             Vec3<Real> const& up = {0, 1, 0}) {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);

  auto const zaxis = normalize(eye - center);
  auto const xaxis = cross(normalize(up), zaxis);
  auto const yaxis = cross(zaxis, xaxis);
  return Mat4<Real>{{xaxis.x(), xaxis.y(), xaxis.z(), -dot(xaxis, eye)},
                    {yaxis.x(), yaxis.y(), yaxis.z(), -dot(yaxis, eye)},
                    {zaxis.x(), zaxis.y(), zaxis.z(), -dot(zaxis, eye)},
                    {O, O, O, I}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr look_at_matrix(Vec3<Real> const& eye, Vec3<Real> const& center,
                              Vec3<Real> const& up = {0, 1, 0}) -> Mat4<Real> {
  return look_at_matrix_right_hand(eye, center, up);
}
//------------------------------------------------------------------------------
/// Can be used as view matrix of a camera.
template <typename Real>
auto constexpr inv_look_at_matrix(Vec3<Real> const& eye,
                                  Vec3<Real> const& center,
                                  Vec3<Real> const& up = {0, 1, 0}) {
  return inv_look_at_matrix_left_hand(eye, center, up);
}
//==============================================================================
/// Can be used as transform matrix of an object.
template <typename Real>
auto constexpr fps_look_at_matrix_left_hand(Vec3<Real> const&     eye,
                                            arithmetic auto const pitch,
                                            arithmetic auto const yaw)
    -> Mat4<Real> {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);

  auto const cos_pitch = gcem::cos(pitch);
  auto const sin_pitch = gcem::sin(pitch);
  auto const cos_yaw   = gcem::cos(yaw);
  auto const sin_yaw   = gcem::sin(yaw);

  auto xaxis = normalize(vec{-cos_yaw, 0, sin_yaw});
  auto yaxis =
      normalize(vec{sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch});
  auto zaxis =
      normalize(vec{-sin_yaw * cos_pitch, sin_pitch, -cos_pitch * cos_yaw});

  return Mat4<Real>{{xaxis.x(), yaxis.x(), zaxis.x(), eye.x()},
                    {xaxis.y(), yaxis.y(), zaxis.y(), eye.y()},
                    {xaxis.z(), yaxis.z(), zaxis.z(), eye.z()},
                    {O, O, O, I}};
}
//------------------------------------------------------------------------------
/// Can be used as view matrix of a camera.
template <typename Real>
auto constexpr inv_fps_look_at_matrix_left_hand(Vec3<Real> const&     eye,
                                                arithmetic auto const pitch,
                                                arithmetic auto const yaw)
    -> Mat4<Real> {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);

  auto const cos_pitch = gcem::cos(pitch);
  auto const sin_pitch = gcem::sin(pitch);
  auto const cos_yaw   = gcem::cos(yaw);
  auto const sin_yaw   = gcem::sin(yaw);

  auto const xaxis = normalize(vec{-cos_yaw, 0, sin_yaw});
  auto const yaxis =
      normalize(vec{sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch});
  auto const zaxis =
      normalize(vec{-sin_yaw * cos_pitch, sin_pitch, -cos_pitch * cos_yaw});

  return Mat4<Real>{{xaxis.x(), xaxis.y(), xaxis.z(), -dot(xaxis, eye)},
                    {yaxis.x(), yaxis.y(), yaxis.z(), -dot(yaxis, eye)},
                    {zaxis.x(), zaxis.y(), zaxis.z(), -dot(zaxis, eye)},
                    {O, O, O, I}};
}
//------------------------------------------------------------------------------
/// Can be used as transform matrix of an object.
template <typename Real>
auto constexpr fps_look_at_matrix_right_hand(Vec3<Real> const&     eye,
                                             arithmetic auto const pitch,
                                             arithmetic auto const yaw)
    -> Mat4<Real> {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);

  auto cos_pitch = gcem::cos(pitch);
  auto sin_pitch = gcem::sin(pitch);
  auto cos_yaw   = gcem::cos(yaw);
  auto sin_yaw   = gcem::sin(yaw);

  auto xaxis = normalize(vec{cos_yaw, 0, -sin_yaw});
  auto yaxis =
      normalize(vec{sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch});
  auto zaxis =
      normalize(vec{sin_yaw * cos_pitch, -sin_pitch, cos_pitch * cos_yaw});

  return Mat4<Real>{{xaxis.x(), yaxis.x(), zaxis.x(), eye.x()},
                    {xaxis.y(), yaxis.y(), zaxis.y(), eye.y()},
                    {xaxis.z(), yaxis.z(), zaxis.z(), eye.z()},
                    {O, O, O, I}};
}
//------------------------------------------------------------------------------
/// Can be used as view matrix of a camera.
template <typename Real>
auto constexpr inv_fps_look_at_matrix_right_hand(Vec3<Real> const&     eye,
                                                 arithmetic auto const pitch,
                                                 arithmetic auto const yaw)
    -> Mat4<Real> {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);

  auto const cos_pitch = gcem::cos(pitch);
  auto const sin_pitch = gcem::sin(pitch);
  auto const cos_yaw   = gcem::cos(yaw);
  auto const sin_yaw   = gcem::sin(yaw);

  auto const xaxis = normalize(vec{cos_yaw, 0, -sin_yaw});
  auto const yaxis =
      normalize(vec{sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch});
  auto const zaxis =
      normalize(vec{sin_yaw * cos_pitch, -sin_pitch, cos_pitch * cos_yaw});

  return Mat4<Real>{{xaxis.x(), xaxis.y(), xaxis.z(), -dot(xaxis, eye)},
                    {yaxis.x(), yaxis.y(), yaxis.z(), -dot(yaxis, eye)},
                    {zaxis.x(), zaxis.y(), zaxis.z(), -dot(zaxis, eye)},
                    {O, O, O, I}};
}
//------------------------------------------------------------------------------
/// Can be used as transform matrix of an object.
template <typename Real>
auto constexpr fps_look_at_matrix(Vec3<Real> const&     eye,
                                  arithmetic auto const pitch,
                                  arithmetic auto const yaw) -> Mat4<Real> {
  return fps_look_at_matrix_right_hand(eye, pitch, yaw);
}
//------------------------------------------------------------------------------
/// Can be used as view matrix of a camera.
template <typename Real>
auto constexpr inv_fps_look_at_matrix(Vec3<Real> const&     eye,
                                      arithmetic auto const pitch,
                                      arithmetic auto const yaw) {
  return inv_fps_look_at_matrix_left_hand(eye, pitch, yaw);
}
//==============================================================================
template <typename Real>
auto constexpr frustum_matrix(Real const left, Real const right,
                              Real const bottom, Real const top,
                              Real const near, Real const far) {
  auto constexpr O = Real(0);
  auto constexpr I = Real(1);

  auto const n2        = 2 * near;
  auto const width     = right - left;
  auto const inv_width = 1 / width;
  auto const xs        = n2 * inv_width;
  auto const xzs       = (right + left) * inv_width;

  auto const height     = top - bottom;
  auto const inv_height = 1 / height;
  auto const ys         = n2 * inv_height;
  auto const yzs        = (top + bottom) * inv_height;

  auto const depth     = far - near;
  auto const inv_depth = 1 / depth;
  auto const zs        = -(far + near) * inv_depth;
  auto const zt        = -2 * near * far * inv_depth;

  return Mat4<Real>{
      {xs,  O, xzs,  O},
      { O, ys, yzs,  O},
      { O,  O,  zs, zt},
      { O,  O,  -I,  O}};
}
//------------------------------------------------------------------------------
template <typename Real>
auto constexpr perspective_matrix(Real const fov_angles, Real const aspect_ratio,
                                  Real const near, Real const far) {
  auto constexpr half              = Real(1) / Real(2);
  auto constexpr angles_to_radians_factor = Real(M_PI) / Real(180);
  auto const fov_radians     = fov_angles * angles_to_radians_factor;
  auto const scale           = gcem::tan(fov_radians * half) * near;
  auto const right           = scale * aspect_ratio;
  auto const left            = -right;
  auto const top             = scale;
  auto const bottom          = -top;
  return frustum_matrix(left, right, bottom, top, near, far);
  //auto constexpr O = Real(0);
  //auto constexpr I = Real(1);
  //auto P           = Mat4<Real>{
  //              {2 * near / (right - left), O, (right + left) / (right - left), O},
  //              {O, 2 * near / (top - bottom), (top + bottom) / (top - bottom), O},
  //              {O, O, (far + near) / (far - near), 2 * far * near / (far - near)},
  //              {O, O, -1, O}};
  //return P;
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
