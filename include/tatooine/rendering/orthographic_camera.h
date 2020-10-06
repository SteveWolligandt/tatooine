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
/// \brief Perspective cameras are able to cast rays from one point called 'eye'
/// through an image plane.
///
/// Based on the eye position, a look-at point and a field of view angle the
/// image plane gets constructed.
/// This camera class constructs a right-handed coordinate system.
template <real_number Real>
class orthographic_camera : public camera<Real> {
 public:
  using real_t   = Real;
  using parent_t = camera<Real>;
  using this_t   = orthographic_camera<Real>;
  using vec3     = vec<Real, 3>;
  using mat4     = mat<Real, 4, 4>;

 private:
  //----------------------------------------------------------------------------
  // class variables
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
  vec3   m_eye, m_lookat, m_up;
  vec3   m_bottom_left;
  vec3   m_plane_base_x, m_plane_base_y;
  Real m_left, m_right, m_bottom, m_top, m_near, m_far;

 public:
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                      Real const left, Real const right,
                      Real const bottom, Real const top,
                      Real const near, Real const far,
                      size_t const res_x, size_t const res_y)
      : parent_t{eye, lookat, up,res_x, res_y},
        m_left{left},
        m_right{right},
        m_bottom{bottom},
        m_top{top},
        m_near{near},
        m_far{far} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat,
                      Real const left, Real const right,
                      Real const bottom, Real const top,
                      Real const near, Real const far,
                      size_t const res_x, size_t const res_y)
      : orthographic_camera(eye, lookat, vec3{0, 1, 0}, left, right, bottom, top, near, far, res_x, res_y) {}
  //----------------------------------------------------------------------------
  ~orthographic_camera() override = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  /// gets a ray through plane at pixel with coordinate [x,y].
  /// [0,0] is bottom left.
  /// ray goes through center of pixel
  tatooine::ray<Real, 3> ray(Real x, Real y) const override {
    assert(x < this->plane_width());
    assert(y < this->plane_height());
    auto const view_plane_point =
        m_bottom_left + x * m_plane_base_x + y * m_plane_base_y;
    return {{m_eye}, {view_plane_point - m_eye}};
  }
  //============================================================================
 private:
  void setup() override {
    vec3 const view_dir          = normalize(m_lookat - m_eye);
    vec3 const u                 = cross(m_up, view_dir);
    vec3 const v                 = cross(view_dir, u);
    Real const plane_half_width  =  (m_top - m_bottom)/this->aspect_ratio();
    Real const plane_half_height = (m_top - m_bottom) / 2;
    m_left = -plane_half_width;
    m_right = plane_half_width;

    m_bottom_left =
        m_eye + view_dir - u * plane_half_width - v * plane_half_height;
    m_plane_base_x = u * 2 * plane_half_width / (this->plane_width() - 1);
    m_plane_base_y = v * 2 * plane_half_height / (this->plane_height() - 1);
  }
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  std::unique_ptr<parent_t> clone() const override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() const -> mat4 override {
    return {{2 / (m_right - m_left), Real(0), Real(0),
             -(m_right + m_left) / (m_right - m_left)},
            {Real(0), 2 / (m_top - m_bottom), Real(0),
             -(m_top + m_bottom) / (m_top - m_bottom)},
            {Real(0), Real(0), -2 / (m_far - m_near),
             -(m_far + m_near) / (m_far - m_near)},
            {Real(0), Real(0), Real(0), Real(1)}};
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
