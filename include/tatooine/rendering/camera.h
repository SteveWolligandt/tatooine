#ifndef TATOOINE_RENDERING_CAMERA_H
#define TATOOINE_RENDERING_CAMERA_H
//==============================================================================
#include <tatooine/clonable.h>
#include <tatooine/concepts.h>
#include <tatooine/ray.h>
#include <tatooine/rendering/matrices.h>
#include <tatooine/vec.h>

#include <array>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
/// \brief Interface for camera implementations.
///
/// Implementations must override the ray method that casts rays through the
/// camera's image plane.
template <floating_point Real, typename Derived>
struct camera_interface {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using real_t = Real;
  using this_t = camera_interface<Real, Derived>;
  using vec3   = Vec3<Real>;
  using vec4   = Vec4<Real>;
  using mat4   = Mat4<Real>;

  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
 private:
  vec3              m_eye, m_lookat, m_up;
  Real              m_near, m_far;
  Vec4<std::size_t> m_viewport;
  vec3              m_bottom_left;
  vec3              m_plane_base_x, m_plane_base_y;

  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
 public:
  camera_interface(vec3 const& eye, vec3 const& lookat, vec3 const& up, Real const near,
         Real const far, std::size_t const res_x, std::size_t const res_y)
      : m_eye{eye},
        m_lookat{lookat},
        m_up{up},
        m_near{near},
        m_far{far},
        m_viewport{0, 0, res_x, res_y} {}
  //----------------------------------------------------------------------------
  ~camera_interface() = default;
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
    return static_cast<Real>(m_viewport(2)) / static_cast<Real>(m_viewport(3));
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
  auto set_viewport(std::size_t const bottom, std::size_t const left,
                    std::size_t const width, std::size_t const height) {
    m_viewport(0) = bottom;
    m_viewport(1) = left;
    m_viewport(2) = width;
    m_viewport(3) = height;
    setup();
  }
  //----------------------------------------------------------------------------
  auto set_resolution(std::size_t const width, std::size_t const height) {
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
  auto n() const { return near(); }
  auto set_near(Real const near) {
    m_near = near;
    setup();
  }
  //----------------------------------------------------------------------------
  auto far() const { return m_far; }
  auto f() const { return far(); }
  auto set_far(Real const far) {
    m_far = far;
    setup();
  }
  //----------------------------------------------------------------------------
  auto depth() const { return f() - n(); }
  auto d() const { return depth(); }
  //----------------------------------------------------------------------------
  auto transform_matrix() const -> mat4 { return *inv(view_matrix()); }
  //----------------------------------------------------------------------------
  auto view_matrix() const -> mat4 {
    return look_at_matrix(m_eye, m_lookat, m_up);
  }
  //----------------------------------------------------------------------------
  /// Projects a screen coordinates to world coordinates.
  auto unproject(vec4 x) const {
    // Transformation of normalized coordinates between -1 and 1
    x(0) = (x(0) - m_viewport(0)) / plane_width() * 2 - 1;
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
  /// Projects a world coordinate to screen coordinates.
  auto project(vec3 const& x) const {
    return project(vec4{x(0), x(1), x(2), 1});
  }
  //----------------------------------------------------------------------------
  /// Projects a homogeneous world coordinate to screen coordinates.
  auto project(vec4 p) const {
    p    = projection_matrix() * view_matrix() * p;
    p(0) = p(0) / p(3);
    p(1) = p(1) / p(3);
    p(2) = p(2) / p(3);
    p(3) = 1;

    // [-1,1] to [0,1]
    p(0) = p(0) * Real(0.5) + Real(0.5);
    p(1) = p(1) * Real(0.5) + Real(0.5);
    p(2) = p(2) * Real(0.5) + Real(0.5);

    // [0,1] to viewport
    p(0) = p(0) * (m_viewport(2) - 1) + m_viewport(0);
    p(1) = p(1) * (m_viewport(3) - 1) + m_viewport(1);

    return p;
  }
  //------------------------------------------------------------------------------
  /// \brief Gets a ray through plane at pixel with coordinate [x,y].
  ///
  /// [0,0] is bottom left.
  /// ray goes through center of pixel.
  auto ray(Real const x, Real const y) const -> tatooine::ray<Real, 3> {
    auto const view_plane_point =
        m_bottom_left + x * m_plane_base_x + y * m_plane_base_y;
    return {{eye()}, {view_plane_point - eye()}};
  }
  //------------------------------------------------------------------------------
  auto setup() -> void {
    auto const A = *inv(projection_matrix() * this->view_matrix());

    auto const bottom_left_homogeneous = (A * Vec4<Real>{-1, -1, -1, 1});
    m_bottom_left = bottom_left_homogeneous.xyz() / bottom_left_homogeneous.w();
    auto const bottom_right = A * Vec4<Real>{1, -1, -1, 1};
    auto const top_left     = A * Vec4<Real>{-1, 1, -1, 1};
    m_plane_base_x = (bottom_right.xyz() / bottom_right.w() - m_bottom_left) /
                     (this->plane_width() - 1);
    m_plane_base_y = (top_left.xyz() / top_left.w() - m_bottom_left) /
                     (this->plane_height() - 1);
  }
  //----------------------------------------------------------------------------
  // interface methods
  //----------------------------------------------------------------------------
  auto projection_matrix() const -> mat4 {
    return static_cast<Derived const*>(this)->projection_matrix();
  };
};
//==============================================================================
namespace details::camera {
//==============================================================================
template <typename Derived, std::floating_point Real>
auto ptr_convertible_to_camera_interface(
    const volatile camera_interface<Real, Derived>*) -> std::true_type;
template <typename>
auto ptr_convertible_to_camera_interface(const volatile void*)
    -> std::false_type;

template <typename>
auto is_derived_from_camera_interface(...) -> std::true_type;
template <typename D>
auto is_derived_from_camera_interface(int) -> decltype(
    ptr_convertible_to_camera_interface<D>(static_cast<D*>(nullptr)));
//==============================================================================
}  // namespace details::camera
//==============================================================================
template <typename T>
struct is_camera_impl
    : std::integral_constant<
          bool,
          std::is_class_v<T>&& decltype(
              details::camera::is_derived_from_camera_interface<T>(0))::value> {
};
//------------------------------------------------------------------------------
template <typename T>
static auto constexpr is_camera = is_camera_impl<T>::value;
//==============================================================================
template <typename T>
concept camera = is_camera<T>;
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
