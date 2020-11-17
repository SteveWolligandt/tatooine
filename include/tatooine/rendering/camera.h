#ifndef TATOOINE_RENDERING_CAMERA_H
#define TATOOINE_RENDERING_CAMERA_H
//==============================================================================
#include <tatooine/clonable.h>
#include <tatooine/concepts.h>
#include <tatooine/ray.h>
#include <tatooine/vec.h>

#include <array>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
/// \brief Interface for camera implementations.
///
/// Implementations must override the ray method that casts rays through the
/// camera's image plane.
template <real_number Real>
struct camera : clonable<camera<Real>> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using this_t            = camera<Real>;
  using real_t            = Real;
  using parent_clonable_t = clonable<camera<Real>>;
  using vec3              = vec<Real, 3>;
  using mat4              = mat<Real, 4, 4>;

  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
 private:
  vec3                  m_eye, m_lookat, m_up;
  std::array<size_t, 2> m_resolution;

  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
 public:
  camera(vec3 const& eye, vec3 const& lookat, vec3 const& up, size_t res_x,
         size_t res_y)
      : m_eye{eye}, m_lookat{lookat}, m_up{up}, m_resolution{res_x, res_y} {}
  virtual ~camera() = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  /// Returns number of pixels of plane in x-direction.
  auto plane_width() const { return m_resolution[0]; }
  //----------------------------------------------------------------------------
  /// Returns number of pixels of plane in y-direction.
  auto plane_height() const { return m_resolution[1]; }
  //----------------------------------------------------------------------------
  auto aspect_ratio() const {
    return static_cast<Real>(m_resolution[0]) /
           static_cast<Real>(m_resolution[1]);
  }
  //----------------------------------------------------------------------------
  auto eye() const -> auto& { return m_eye; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_eye(Real const x, Real const y, Real const z) -> void {
    m_eye = {x, y, z};
    setup();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_eye(vec3 const& eye) -> void {
    m_eye = eye;
    setup();
  }
  //----------------------------------------------------------------------------
  auto lookat() const -> auto& { return m_lookat; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_lookat(Real const x, Real const y, Real const z) -> void {
    m_lookat = {x, y, z};
    setup();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_lookat(vec3 const lookat) -> void {
    m_lookat = lookat;
    setup();
  }
  //----------------------------------------------------------------------------
  auto up() const -> auto& { return m_up; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_up(Real const x, Real const y, Real const z) -> void {
    m_up = {x, y, z};
    setup();
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto set_up(vec3 const up) -> void {
    m_up = up;
    setup();
  }
  auto set_resolution(size_t width, size_t height) {
    m_resolution[0] = width;
    m_resolution[1] = height;
    setup();
  }
  //----------------------------------------------------------------------------
  auto look_at(vec3 const& eye, vec3 const& lookat, vec3 const& up = {0, 1, 0})
      -> void {
    m_eye    = eye;
    m_lookat = lookat;
    m_up     = up;
    setup();
  }
  //----------------------------------------------------------------------------
  auto transform_matrix() const -> mat4 {
    return look_at_matrix(m_eye, m_lookat, m_up);
  }
  //----------------------------------------------------------------------------
  auto view_matrix() const -> mat4 { return inv(transform_matrix()); }
  //----------------------------------------------------------------------------
  // interface methods
  //----------------------------------------------------------------------------
  /// \brief Gets a ray through plane at pixel with coordinate [x,y].
  ///
  /// [0,0] is bottom left.
  /// ray goes through center of pixel.
  /// This method must be overridden in camera implementations.
  virtual auto setup() -> void                                     = 0;
  virtual auto ray(Real x, Real y) const -> tatooine::ray<Real, 3> = 0;
  virtual auto projection_matrix() const -> mat4                   = 0;
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
