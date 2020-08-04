#ifndef TATOOINE_PERSPECTIVE_CAMERA_H
#define TATOOINE_PERSPECTIVE_CAMERA_H
//==============================================================================
#include <tatooine/camera.h>
#include <tatooine/ray.h>
#include <tatooine/vec.h>

#include <cassert>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \brief Perspective cameras are able to cast rays from one point called 'eye'
/// through an image plane.
///
/// Based on the eye position, a look-at point and a field of view angle the
/// image plane gets constructed.
/// This camera class constructs a right-handed coordinate system.
template <real_number Real>
class perspective_camera : public camera<Real> {
 public:
  using real_t   = Real;
  using parent_t = camera<real_t>;
  using this_t   = perspective_camera<real_t>;
  using vec3_t   = vec<real_t, 3>;

 private:
  //----------------------------------------------------------------------------
  // class variables
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
  vec3_t m_eye, m_lookat, m_up;
  vec3_t m_bottom_left;
  vec3_t m_plane_base_x, m_plane_base_y;

 public:
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  perspective_camera(vec3_t const& eye, vec3_t const& lookat, vec3_t const& up,
                     real_t const fov, size_t const res_x, size_t const res_y)
      : parent_t{res_x, res_y}, m_eye{eye}, m_lookat{lookat}, m_up{up} {
    vec3_t const view_dir          = normalize(m_lookat - m_eye);
    vec3_t const u                 = cross(m_up, view_dir);
    vec3_t const v                 = cross(view_dir, u);
    real_t const plane_half_width  = std::tan(fov / real_t(2) * real_t(M_PI) / real_t(180));
    real_t const plane_half_height = static_cast<real_t>(res_y) /
                                     static_cast<real_t>(res_x) *
                                     plane_half_width;
    m_bottom_left =
        m_eye + view_dir - u * plane_half_width - v * plane_half_height;
    m_plane_base_x = u * 2 * plane_half_width / (res_x - 1);
    m_plane_base_y = v * 2 * plane_half_height / (res_y - 1);
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  perspective_camera(vec3_t const& eye, vec3_t const& lookat, real_t fov,
                     size_t res_x, size_t res_y)
      : perspective_camera(eye, lookat, vec3_t{0, 1, 0}, fov, res_x, res_y) {}
  //----------------------------------------------------------------------------
  ~perspective_camera() override = default;
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
  //----------------------------------------------------------------------------
  std::unique_ptr<parent_t> clone() const override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
