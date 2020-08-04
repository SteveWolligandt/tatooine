#ifndef CG_PERSPECTIVE_CAMERA_H
#define CG_PERSPECTIVE_CAMERA_H
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
  using parent_t = camera<Real>;
  using this_t   = perspective_camera<Real>;
  using vec3     = vec<Real, 3>;

 private:
  //----------------------------------------------------------------------------
  // class variables
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
  vec3       m_eye, m_up, m_u, m_v;
  const Real m_plane_half_width, m_plane_half_height;
  const vec3 m_bottom_left;
  const vec3 m_plane_base_x, m_plane_base_y;

 public:
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  perspective_camera(const vec3& eye, const vec3& lookat, const vec3& up,
                     Real fov, size_t res_x, size_t res_y)
      : parent_t{res_x, res_y},
        m_eye{eye},
        m_up{up},
        m_u{cross(normalize(lookat - eye), up)},
        m_v{cross(m_u, normalize(lookat - eye))},
        m_plane_half_width{std::tan(fov * M_PI / 180)},
        m_plane_half_height{double(res_y) / double(res_x) * m_plane_half_width},
        m_bottom_left{lookat - m_u * m_plane_half_width -
                      m_v * m_plane_half_height},
        m_plane_base_x{m_u * 2 * m_plane_half_width / res_x},
        m_plane_base_y{m_v * 2 * m_plane_half_height / res_y} {}
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  perspective_camera(const vec3& eye, const vec3& lookat,  Real fov,
                     size_t res_x, size_t res_y)
    :perspective_camera(eye, lookat, vec3{0,1,0}, fov, res_x, res_y){}
  //----------------------------------------------------------------------------
  ~perspective_camera() override = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  /// gets a ray through plane at pixel with coordinate [x,y].
  /// [0,0] is bottom left.
  /// ray goes through center of pixel
  tatooine::ray<Real, 3> ray(Real x, Real y) const override {
    assert(x < this->plane_width());
    assert(y < this->plane_height());
    const auto view_plane_point =
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
