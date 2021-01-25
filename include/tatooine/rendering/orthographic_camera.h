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
template <arithmetic Real>
class orthographic_camera : public camera<Real> {
 public:
  using real_t   = Real;
  using parent_t = camera<Real>;
  using this_t   = orthographic_camera<Real>;
  using vec3     = vec<Real, 3>;
  using mat4     = mat<Real, 4, 4>;

 private:
  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
  vec3 m_bottom_left;
  vec3 m_plane_base_x, m_plane_base_y;
  Real m_height, m_near, m_far;

 public:
  //----------------------------------------------------------------------------
  // getter / setter
  //----------------------------------------------------------------------------
  auto height() const {
    return m_height;
  }
  auto near() const {
    return m_near;
  }
  auto far() const {
    return m_far;
  }
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                      Real const height, Real const near, Real const far,
                      size_t const res_x, size_t const res_y)
      : parent_t{eye, lookat, up, res_x, res_y},
        m_height{height},
        m_near{near},
        m_far{far} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, Real const height,
                      Real const near, Real const far, size_t const res_x,
                      size_t const res_y)
      : orthographic_camera(eye, lookat, vec3{0, 1, 0}, height, near, far,
                            res_x, res_y) {}
  //----------------------------------------------------------------------------
  ~orthographic_camera() override = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  /// gets a ray through plane at pixel with coordinate [x,y].
  /// [0,0] is bottom left.
  /// ray goes through center of pixel
  auto ray(Real x, Real y) const -> tatooine::ray<Real, 3> override {
    assert(x < this->plane_width());
    assert(y < this->plane_height());
    auto const view_plane_point =
        m_bottom_left + x * m_plane_base_x + y * m_plane_base_y;
    return {view_plane_point, normalize(this->lookat() - this->eye())};
  }
  //============================================================================
 private:
  auto setup() -> void override {
    vec3 const view_dir          = normalize(this->lookat() - this->eye());
    vec3 const u                 = cross(view_dir, this->up());
    vec3 const v                 = cross(u, view_dir);
    Real const plane_half_height = m_height / 2;
    Real const plane_half_width  = plane_half_height * this->aspect_ratio();

    m_bottom_left = this->eye() + view_dir * m_near
                                - u * plane_half_width
                                - v * plane_half_height;
    m_plane_base_x = u * 2 * plane_half_width / (this->plane_width() - 1);
    m_plane_base_y = v * 2 * plane_half_height / (this->plane_height() - 1);
  }
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  auto clone() const -> std::unique_ptr<parent_t> override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto projection_matrix() const -> mat4 override {
    Real const width  = m_height * this->aspect_ratio();
    return {{2 / width, Real(0), Real(0), Real(0)},
            {Real(0), 2 / m_height, Real(0), Real(0)},
            {Real(0), Real(0), -2 / (m_far - m_near),
             -(m_far + m_near) / (m_far - m_near)},
            {Real(0), Real(0), Real(0), Real(1)}};
  }
  auto setup(vec3 const& eye, vec3 const& lookat, vec3 const& up,
             Real const height, Real const near, Real const far,
             size_t const res_x, size_t const res_y) -> void {
    this->set_eye(eye);
    this->set_lookat(lookat);
    this->set_up(up);
    this->set_resolution(res_x, res_y);
    m_height = height;
    m_near   = near;
    m_far    = far;
    setup();
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
