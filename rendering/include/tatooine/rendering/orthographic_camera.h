#ifndef TATOOINE_RENDERING_ORTHOGRAPHIC_CAMERA_H
#define TATOOINE_RENDERING_ORTHOGRAPHIC_CAMERA_H
//==============================================================================
#include <tatooine/ray.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/matrices.h>
#include <tatooine/vec.h>

#include <cassert>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <arithmetic Real>
struct orthographic_camera : camera_interface<Real> {
 public:
  using real_type   = Real;
  using this_type   = orthographic_camera<Real>;
  using parent_type = camera_interface<Real>;
  using vec3        = vec<Real, 3>;
  using mat4        = mat<Real, 4, 4>;
  using parent_type::set_projection_matrix;
  using parent_type::viewport;

 public:
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                vec3 const& up, Real const left,
                                Real const right, Real const bottom,
                                Real const top, Real const near, Real const far,
                                Vec4<std::size_t> const& viewport)
      : parent_type{eye,
                    lookat,
                    up,
                    viewport,
                    orthographic_matrix(left, right, bottom, top, near, far)} {
    // setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye,
                                vec3 const& lookat,
                                vec3 const& up,
                                Real const left, Real const right,
                                Real const bottom, Real const top,
                                Real const near, Real const far,
                                std::size_t const res_x,
                                std::size_t const res_y)
      : parent_type{eye,
                    lookat,
                    up,
                    Vec4<std::size_t>{0, 0, res_x, res_y},
                    orthographic_matrix(left, right, bottom, top, near, far)} {
    // setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                Real const left, Real const right,
                                Real const bottom, Real const top,
                                Real const near, Real const far,
                                Vec4<std::size_t> const& viewport)
      : parent_type{eye,
                    lookat,
                    vec3{0, 1, 0},
                    viewport,
                    orthographic_matrix(left, right, bottom, top, near, far)} {
    // setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                Real const left, Real const right,
                                Real const bottom, Real const top,
                                Real const near, Real const far,
                                std::size_t const res_x,
                                std::size_t const res_y)
      : parent_type{eye,
                    lookat,
                    vec3{0, 1, 0},
                    Vec4<std::size_t>{0, 0, res_x, res_y},
                    orthographic_matrix(left, right, bottom, top, near, far)} {
    // setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                vec3 const& up, Real const height,
                                Real const near, Real const far,
                                Vec4<std::size_t> const& viewport)
      : orthographic_camera{eye,
                            lookat,
                            up,
                            -height / 2 * Real(viewport[2]) * Real(viewport[3]),
                            height / 2 * Real(viewport[2]) * Real(viewport[3]),
                            -height / 2,
                            height / 2,
                            near,
                            far,
                            viewport} {
    // setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                vec3 const& up, Real const height,
                                Real const near, Real const far,
                                std::size_t const res_x,
                                std::size_t const res_y)
      : orthographic_camera{eye,
                            lookat,
                            up,
                            -height / 2 * Real(res_x) / Real(res_y),
                            height / 2 * Real(res_x) / Real(res_y),
                            -height / 2,
                            height / 2,
                            near,
                            far,
                            res_x,
                            res_y} {
    // setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, Real const height,
                      Real const near, Real const far,
                      Vec4<std::size_t> const& viewport)
      : orthographic_camera{eye,  lookat, vec3{0, 1, 0}, height,
                            near, far,    viewport} {}
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, Real const height,
                      Real const near, Real const far, std::size_t const res_x,
                      std::size_t const res_y)
      : orthographic_camera{eye,
                            lookat,
                            vec3{0, 1, 0},
                            height,
                            near,
                            far,
                            Vec4<std::size_t>{0, 0, res_x, res_y}} {}
  //----------------------------------------------------------------------------
  ~orthographic_camera() = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  auto constexpr set_projection_matrix(Real const left, Real const right,
                                       Real const bottom, Real const top,
                                       Real const near, Real const far) {
    set_projection_matrix(
        orthographic_matrix(left, right, bottom, top, near, far));
  }
  //----------------------------------------------------------------------------
  auto constexpr set_projection_matrix(Real const height, Real const near = 100,
                                       Real const far = -100) {
    set_projection_matrix(-height / 2 * this->aspect_ratio(),
                          height / 2 * this->aspect_ratio(), -height / 2, height / 2,
                          near, far);
  }
  //----------------------------------------------------------------------------
  auto width() const { return 2 / this->projection_matrix()(0, 0); }
  auto height() const { return 2 / this->projection_matrix()(1, 1); }
  auto depth() const { return -2 / this->projection_matrix()(2, 2); }
  //----------------------------------------------------------------------------
  // auto constexpr setup(vec3 const& eye, vec3 const& lookat, vec3 const& up,
  //                     Real const width, Real const height, Real const near,
  //                     Real const far, std::size_t const res_x,
  //                     std::size_t const res_y) -> void {
  //  this->set_eye_without_update(eye);
  //  this->set_lookat_without_update(lookat);
  //  this->set_up_without_update(up);
  //  this->set_resolution_without_update(res_x, res_y);
  //  m_top    = height / 2;
  //  m_bottom = -m_top;
  //  m_right  = width / 2;
  //  m_left   = -m_right;
  //  this->set_near_without_update(near);
  //  this->set_far_without_update(far);
  //  setup();
  //}
  ////----------------------------------------------------------------------------
  // auto constexpr setup(Real const width, Real const height) -> void {
  //   m_top    = height / 2;
  //   m_bottom = -m_top;
  //   m_right  = width / 2;
  //   m_left   = -m_right;
  //   setup();
  // }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
