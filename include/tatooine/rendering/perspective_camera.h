#ifndef TATOOINE_RENDERING_PERSPECTIVE_CAMERA_H
#define TATOOINE_RENDERING_PERSPECTIVE_CAMERA_H
//==============================================================================
#include <tatooine/ray.h>
#include <tatooine/rendering/camera.h>
#include <tatooine/rendering/matrices.h>

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
template <arithmetic Real>
class perspective_camera : public camera<Real> {
 public:
  using real_t   = Real;
  using parent_t = camera<Real>;
  using this_t   = perspective_camera<Real>;
  using parent_t::eye;
  using parent_t::lookat;
  using parent_t::up;
  using typename parent_t::mat4;
  using typename parent_t::vec3;

 private:
  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
  vec3   m_bottom_left;
  vec3   m_plane_base_x, m_plane_base_y;
  Real m_fov, m_near, m_far;

 public:
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  perspective_camera(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                     Real const fov, Real const near, Real const far,
                     size_t const res_x, size_t const res_y)
      : parent_t{eye, lookat, up, res_x, res_y},
        m_fov{fov},
        m_near{near},
        m_far{far} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  perspective_camera(vec3 const& eye, vec3 const& lookat, Real fov,
                     Real const near, Real const far, size_t const res_x,
                     size_t const res_y)
      : perspective_camera(eye, lookat, vec3{0, 1, 0}, fov, near, far, res_x,
                           res_y) {}
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  perspective_camera(vec3 const& eye, vec3 const& lookat, Real fov,
                     size_t const res_x, size_t const res_y)
      : perspective_camera(eye, lookat, vec3{0, 1, 0}, fov, 0.001, 1000, res_x,
                           res_y) {}
  //----------------------------------------------------------------------------
  ~perspective_camera() override = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  /// gets a ray through plane at pixel with coordinate [x,y].
  /// [0,0] is bottom left.
  /// ray goes through center of pixel
  auto ray(Real const x, Real const y) const -> tatooine::ray<Real, 3> override {
    auto const view_plane_point =
        m_bottom_left + x * m_plane_base_x + y * m_plane_base_y;
    return {{eye()}, {view_plane_point - eye()}};
  }
  //============================================================================
 private:
  void setup() override {
    vec3 const   view_dir = normalize(lookat() - eye());
    vec3 const   u        = cross(up(), view_dir);
    vec3 const   v        = cross(view_dir, u);
    Real const plane_half_width =
        std::tan(m_fov / Real(2) * Real(M_PI) / Real(180)) ;
    Real const plane_half_height = plane_half_width / this->aspect_ratio();
    m_bottom_left =
        eye() + view_dir - u * plane_half_width - v * plane_half_height;
    m_plane_base_x = u * 2 * plane_half_width / (this->plane_width() - 1);
    m_plane_base_y = v * 2 * plane_half_height / (this->plane_height() - 1);
  }
  //----------------------------------------------------------------------------
 public:
  std::unique_ptr<parent_t> clone() const override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() const -> mat4 override {
    Real const plane_half_width =
        std::tan(m_fov / Real(2) * Real(M_PI) / Real(180)) * m_near;
    Real const r = this->aspect_ratio() * plane_half_width;
    Real const l = -r;
    Real const t = plane_half_width;
    Real const b = -t;
    return {{2 * m_near / (r - l), Real(0), (r + l) / (r - l), Real(0)},
            {Real(0), 2 * m_near / (t - b), (t + b) / (t - b), Real(0)},
            {Real(0), Real(0), -(m_far + m_near) / (m_far - m_near),
             -2 * m_far * m_near / (m_far - m_near)},
            {Real(0), Real(0), Real(-1), Real(0)}};
  }
  //----------------------------------------------------------------------------
  void set_fov(Real const fov) {
    m_fov = fov;
    setup();
  }
  //----------------------------------------------------------------------------
  void set_near(Real const near) {
    m_near = near;
    setup();
  }
  //----------------------------------------------------------------------------
  void set_far(Real const far) {
    m_far = far;
    setup();
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
