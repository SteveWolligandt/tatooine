#ifndef TATOOINE_RENDERING_CAMERA_H
#define TATOOINE_RENDERING_CAMERA_H
//==============================================================================
#include <tatooine/clonable.h>
#include <tatooine/concepts.h>
#include <tatooine/ray.h>
#include <tatooine/vec.h>
#include <tatooine/rendering/matrices.h>

#include <array>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
/// \brief Interface for camera implementations.
///
/// Implementations must override the ray method that casts rays through the
/// camera's image plane.
template <floating_point Real>
struct camera : clonable<camera<Real>> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using real_t = Real;
  using this_t = camera<Real>;
  using vec3   = Vec3<Real>;
  using vec4   = Vec4<Real>;
  using mat4   = Mat4<Real>;

  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
 private:
  vec3         m_eye, m_lookat, m_up;
  Real         m_near, m_far;
  Vec4<size_t> m_viewport;

  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
 public:
  camera(vec3 const& eye, vec3 const& lookat, vec3 const& up, Real near,
         Real far, size_t res_x, size_t res_y)
      : m_eye{eye},
        m_lookat{lookat},
        m_up{up},
        m_near{near},
        m_far{far},
        m_viewport{0, 0, res_x, res_x} {}
  virtual ~camera() = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  /// Returns number of pixels of plane in x-direction.
  auto plane_width() const { return m_viewport(2); }
  //----------------------------------------------------------------------------
  /// Returns number of pixels of plane in y-direction.
  auto plane_height() const { return m_viewport(3); }
  //----------------------------------------------------------------------------
  auto aspect_ratio() const {
    return static_cast<Real>(plane_width()) / static_cast<Real>(plane_height());
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
  //----------------------------------------------------------------------------
  auto set_viewport(size_t bottom, size_t left, size_t width, size_t height) {
    m_viewport(0) = bottom;
    m_viewport(1) = left;
    m_viewport(2) = width;
    m_viewport(3) = height;
    setup();
  }
  //----------------------------------------------------------------------------
  auto set_resolution(size_t width, size_t height) {
    m_viewport(2) = width;
    m_viewport(3) = height;
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
  auto near() const { return m_near; }
  auto n() const {return near();}
  auto set_near(Real const near) {
    m_near = near;
    setup();
  }
  //----------------------------------------------------------------------------
  auto far() const { return m_far; }
  auto f() const {return far();}
  auto set_far(Real const far) {
    m_far = far;
    setup();
  }
  //----------------------------------------------------------------------------
  auto depth() const { return f() - n(); }
  auto d() const { return depth(); }
  //----------------------------------------------------------------------------
  auto transform_matrix() const -> mat4 {
    return look_at_matrix(m_eye, m_lookat, m_up);
  }
  //----------------------------------------------------------------------------
  auto view_matrix() const -> mat4 { return *inv(transform_matrix()); }
  //----------------------------------------------------------------------------
  auto unproject(vec4 x) const {
    // Transformation of normalized coordinates between -1 and 1
    x(0) = (x(0) - m_viewport(0)) / m_viewport(2) * 2 - 1;
    x(1) = (m_viewport(3) - x(1) - m_viewport(1)) / m_viewport(3) * 2 - 1;
    x(2) = 2 * x(2) - 1;
    x(3) = 1;
    x    = solve(projection_matrix() * view_matrix(), x);
    x(3) = 1.0 / x(3);
    x(0) = x(0) * x(3);
    x(1) = x(1) * x(3);
    x(2) = x(2) * x(3);
    return x;
  }
  //----------------------------------------------------------------------------
  auto project(vec3 const& x) const {
    return project(vec4{x(0), x(1), x(2), 1});
  }
  //----------------------------------------------------------------------------
  auto project(vec4 x) const {
    x(0) = x(0) / x(3);
    x(1) = x(1) / x(3);
    x(2) = x(2) / x(3);
    x(3) = 1;
    auto const V = view_matrix();
    auto const P = projection_matrix();
    x            = V * x;
    x            = P * x;

    // [-1,1] to [0,1]
    x(0) = (x(0) + 1) / 2;
    x(1) = (x(1) + 1) / 2;
    x(2) = (x(2) + 1) / 2;

    // [0,1] to viewport
    x(0) = x(0) * (m_viewport(2) - 1) + m_viewport(0);
    x(1) = x(1) * (m_viewport(3) - 1) + m_viewport(1);
    return x;
  }
  //----------------------------------------------------------------------------
  // interface methods
  //----------------------------------------------------------------------------
  virtual auto setup() -> void = 0;
  /// \brief Gets a ray through plane at pixel with coordinate [x,y].
  ///
  /// [0,0] is bottom left.
  /// ray goes through center of pixel.
  /// This method must be overridden in camera implementations.
  virtual auto ray(Real x, Real y) const -> tatooine::ray<Real, 3> = 0;
  virtual auto projection_matrix() const -> mat4                   = 0;
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
