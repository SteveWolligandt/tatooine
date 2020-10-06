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
  using parent_t = camera<real_t>;
  using this_t   = orthographic_camera<real_t>;
  using vec3     = vec<real_t, 3>;
  using mat4     = mat<real_t, 4, 4>;

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
  real_t m_left, m_right, m_bottom, m_top, m_near, m_far;

 public:
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                      real_t const left, real_t const right,
                      real_t const bottom, real_t const top,
                      real_t const near, real_t const far,
                      size_t const res_x, size_t const res_y)
      : parent_t{res_x, res_y},
        m_eye{eye},
        m_lookat{lookat},
        m_up{up},
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
                      real_t const left, real_t const right,
                      real_t const bottom, real_t const top,
                      real_t const near, real_t const far,
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
  tatooine::ray<real_t, 3> ray(real_t x, real_t y) const override {
    assert(x < this->plane_width());
    assert(y < this->plane_height());
    auto const view_plane_point =
        m_bottom_left + x * m_plane_base_x + y * m_plane_base_y;
    return {{m_eye}, {view_plane_point - m_eye}};
  }
  //============================================================================
 private:
  void setup() {
    vec3 const   view_dir = normalize(m_lookat - m_eye);
    vec3 const   u        = cross(m_up, view_dir);
    vec3 const   v        = cross(view_dir, u);
    real_t const plane_half_width =
        std::tan(m_fov / real_t(2) * real_t(M_PI) / real_t(180));
    real_t const plane_half_height = plane_half_width * this->aspect_ratio();
    m_bottom_left =
        m_eye + view_dir - u * plane_half_width - v * plane_half_height;
    m_plane_base_x = u * 2 * plane_half_width / (this->plane_width() - 1);
    m_plane_base_y = v * 2 * plane_half_height / (this->plane_height() - 1);
  }
  //----------------------------------------------------------------------------
 public:
  void set_eye(vec3 const& eye) {
    m_eye = eye;
    setup();
  }
  //------------------------------------------------------------------------------
  void set_resolution(size_t const res_x, size_t const res_y) {
    parent_t::set_resolution(res_x, res_y);
    setup();
  }
  //------------------------------------------------------------------------------
  void set_lookat(vec3 const lookat) {
    m_lookat = lookat;
    setup();
  }
  //------------------------------------------------------------------------------
  void set_up(vec3 const up) {
    m_up = up;
    setup();
  }
  //----------------------------------------------------------------------------
  void look_at(vec3 const& eye, vec3 const& lookat,
               vec3 const& up = {0, 1, 0}) {
    m_eye    = eye;
    m_lookat = lookat;
    m_up     = up;
    setup();
  }
  //------------------------------------------------------------------------------
  void look_at(vec3 const& eye, vec3 const& lookat, vec3 const& up,
               real_t const left, real_t const right, real_t const bottom,
               real_t const top, real_t const near, real_t const far) {
    m_eye    = eye;
    m_lookat = lookat;
    m_up     = up;
    m_left   = left;
    m_right  = right;
    m_bottom = bottom;
    m_top    = top;
    m_near  = near;
    m_far   = far;
    setup();
  }
  //----------------------------------------------------------------------------
  void setup(vec3 const& eye, vec3 const& lookat, vec3 const& up,
             real_t const left, real_t const right, real_t const bottom,
             real_t const top, real_t const near, real_t const far,
             size_t res_x, size_t res_y) {
    this->set_resolution(res_x, res_y);
    m_eye    = eye;
    m_lookat = lookat;
    m_up     = up;
    m_left   = left;
    m_right  = right;
    m_bottom = bottom;
    m_top    = top;
    m_near  = near;
    m_far   = far;
    setup();
  }
  //----------------------------------------------------------------------------
  std::unique_ptr<parent_t> clone() const override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() const -> mat4 {
  return {{2 / (m_right - m_left), Real(0), Real(0), -(m_right + m_left) / (m_right - m_left)},
          {Real(0), 2 / (m_top - m_bottom), Real(0), -(m_top + m_bottom) / (m_top - m_bottom)},
          {Real(0), Real(0), -2 / (m_far - m_near), -(m_far + m_near) / (m_far - m_near)},
          {Real(0), Real(0), Real(0), Real(1)}};
  }
  //----------------------------------------------------------------------------
  auto transform_matrix(real_t const near, real_t const far) const
      -> mat4 {
    return look_at_matrix(m_eye, m_lookat, m_up);
  }
  //----------------------------------------------------------------------------
  auto view_matrix(real_t const near, real_t const far) const -> mat4 {
    return inv(transform_matrix(near, far));
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
