#ifndef TATOOINE_RENDERING_CAMERA_H
#define TATOOINE_RENDERING_CAMERA_H
//==============================================================================
#include <tatooine/clonable.h>
#include <tatooine/concepts.h>
#include <tatooine/ray.h>
#include <tatooine/rendering/matrices.h>
#include <tatooine/vec.h>
#include <tatooine/gl/glfunctions.h>

#include <array>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
namespace polymorphic {
template <floating_point Real>
struct camera {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
  using real_type = Real;
  using this_type = camera<Real>;
  using vec2      = Vec2<Real>;
  using vec3      = Vec3<Real>;
  using vec4      = Vec4<Real>;
  using mat4      = Mat4<Real>;

  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
 private:
  Vec4<std::size_t> m_viewport;
  vec3              m_bottom_left;
  vec3              m_plane_base_x, m_plane_base_y;

  mat4              m_transform_matrix;
  mat4              m_projection_matrix;

  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
 public:
  constexpr camera(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                   Vec4<std::size_t> const& viewport)
      : m_viewport{viewport},
        m_transform_matrix{look_at_matrix(eye, lookat, up)},
        m_projection_matrix{mat4::eye()} {}
  //----------------------------------------------------------------------------
  constexpr camera(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                   Vec4<std::size_t> const& viewport, mat4 const& p)
      : m_viewport{viewport},
        m_transform_matrix{look_at_matrix(eye, lookat, up)},
        m_projection_matrix{p} {}
  //----------------------------------------------------------------------------
  virtual ~camera() = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  /// Returns number of pixels of plane in x-direction.
  auto constexpr plane_width() const { return m_viewport(2); }
  //----------------------------------------------------------------------------
  /// Returns number of pixels of plane in y-direction.
  auto constexpr plane_height() const { return m_viewport(3); }
  //----------------------------------------------------------------------------
  auto constexpr aspect_ratio() const {
    return static_cast<Real>(m_viewport(2)) / static_cast<Real>(m_viewport(3));
  }
  //----------------------------------------------------------------------------
  auto constexpr eye() const -> auto {
    return vec3{m_transform_matrix(0, 3),
                m_transform_matrix(1, 3),
                m_transform_matrix(2, 3)};
  }
  //----------------------------------------------------------------------------
  auto constexpr right_direction() const {
    return vec3{m_transform_matrix(0, 0),
                m_transform_matrix(1, 0),
                m_transform_matrix(2, 0)};
  }
  //----------------------------------------------------------------------------
  auto constexpr up_direction() const {
    return vec3{m_transform_matrix(0, 1),
                m_transform_matrix(1, 1),
                m_transform_matrix(2, 1)};
  }
  //----------------------------------------------------------------------------
  auto constexpr view_direction() const {
    return vec3{m_transform_matrix(0, 2),
                m_transform_matrix(1, 2),
                m_transform_matrix(2, 2)};
  }
  //----------------------------------------------------------------------------
  auto constexpr set_viewport_without_update(std::size_t const bottom,
                                             std::size_t const left,
                                             std::size_t const width,
                                             std::size_t const height) {
    m_viewport(0) = bottom;
    m_viewport(1) = left;
    m_viewport(2) = width;
    m_viewport(3) = height;
  }
  //----------------------------------------------------------------------------
  auto constexpr set_viewport(std::size_t const bottom, std::size_t const left,
                              std::size_t const width,
                              std::size_t const height) {
    m_viewport(0) = bottom;
    m_viewport(1) = left;
    m_viewport(2) = width;
    m_viewport(3) = height;
    //setup();
  }
  //----------------------------------------------------------------------------
  auto constexpr set_resolution_without_update(std::size_t const width,
                                               std::size_t const height) {
    m_viewport(2) = width;
    m_viewport(3) = height;
  }
  //----------------------------------------------------------------------------
  auto constexpr set_resolution(std::size_t const width,
                                std::size_t const height) {
    m_viewport(2) = width;
    m_viewport(3) = height;
    //setup();
  }
  //----------------------------------------------------------------------------
  auto set_gl_viewport() const {
    gl::viewport(m_viewport[0], m_viewport[1], m_viewport[2], m_viewport[3]);
  }
  //----------------------------------------------------------------------------
  auto constexpr look_at(vec3 const& eye, vec3 const& lookat,
                         vec3 const& up = {0, 1, 0}) -> void {
    m_transform_matrix = look_at_matrix(eye, lookat, up);
    //setup();
  }
  //----------------------------------------------------------------------------
  auto constexpr look_at(vec3 const& eye, arithmetic auto const pitch,
                         arithmetic auto const yaw) -> void {
    m_transform_matrix = fps_look_at_matrix(eye, pitch, yaw);
    //setup();
  }
  //----------------------------------------------------------------------------
  auto constexpr transform_matrix() const -> auto const& {
    return m_transform_matrix;
  }
  //----------------------------------------------------------------------------
  auto constexpr view_matrix() const   {
    auto const& T = transform_matrix();
    auto const  eye_o_x =
        T(0, 3) * T(0, 0) + T(1, 3) * T(1, 0) + T(2, 3) * T(2, 0);
    auto const eye_o_y =
        T(0, 3) * T(0, 1) + T(1, 3) * T(1, 1) + T(2, 3) * T(2, 1);
    auto const eye_o_z =
        T(0, 3) * T(0, 2) + T(1, 3) * T(1, 2) + T(2, 3) * T(2, 2);
    return mat4{{T(0, 0), T(1, 0), T(2, 0), -eye_o_x},
                {T(0, 1), T(1, 1), T(2, 1), -eye_o_y},
                {T(0, 2), T(1, 2), T(2, 2), -eye_o_z},
                {real_type(0), real_type(0), real_type(0), real_type(1)}};
  }
  //----------------------------------------------------------------------------
  auto constexpr projection_matrix() const -> auto const& {
    return m_projection_matrix;
  }
  //----------------------------------------------------------------------------
  auto view_projection_matrix() const {
    return projection_matrix() * view_matrix();
  }
  //----------------------------------------------------------------------------
  /// Projects a screen coordinates to world coordinates.
  auto unproject(vec2 const& p) const {
    return unproject(vec4{p.x(), p.y(), 0.5, 1});
  }
  //----------------------------------------------------------------------------
  /// Projects a screen coordinates to world coordinates.
  auto unproject(vec3 const& p) const {
    return unproject(vec4{p.x(), p.y(), p.z(), 1});
  }
  //----------------------------------------------------------------------------
  /// Projects a homogeneous screen coordinates to world coordinates.
  auto unproject(vec4 p) const {
    // [0,w-1] x [0,h-1] -> [-1,1] x [-1,1]
    p(0) = (p(0) - m_viewport(0)) / (m_viewport(2) - 1) * 2 - 1;
    p(1) = (p(1) - m_viewport(1)) / (m_viewport(3) - 1) * 2 - 1;
    p(2) = p(2) * 2 - 1;
    p(3) = 1;

    // canonical view volume to world coordinate
    p    = *inv(view_projection_matrix()) * p;
    p(3) = 1 / p(3);
    p(0) = p(0) * p(3);
    p(1) = p(1) * p(3);
    p(2) = p(2) * p(3);
    p(3) = 1;
    return p;
  }
  //----------------------------------------------------------------------------
  /// Projects a world coordinate to screen coordinates.
  auto project(vec2 const& p) const {
    return project(vec4{p(0), p(1), 0, 1});
  }
  //----------------------------------------------------------------------------
  /// Projects a world coordinate to screen coordinates.
  auto project(vec3 const& p) const {
    return project(vec4{p(0), p(1), p(2), 1});
  }
  //----------------------------------------------------------------------------
  /// Projects a homogeneous world coordinate to screen coordinates.
  auto project(vec4 p) const {
    p = view_projection_matrix() * p;
    p(0) /= p(3);

    // [-1,1] -> [0,1]
    p(0) = p(0) * Real(0.5) + Real(0.5);
    p(1) = p(1) * Real(0.5) + Real(0.5);
    p(2) = p(2) * Real(0.5) + Real(0.5);

    // [0,1] to viewport
    p(0) = p(0) * (plane_width() - 1) + m_viewport(0);
    p(1) = p(1) * (plane_height() - 1) + m_viewport(1);

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
  ////------------------------------------------------------------------------------
  //auto setup() -> void {
  //  auto const A = *inv(view_projection_matrix());
  //
  //  auto const bottom_left_homogeneous = (A * Vec4<Real>{-1, -1, -1, 1});
  //  m_bottom_left = bottom_left_homogeneous.xyz() / bottom_left_homogeneous.w();
  //  auto const bottom_right = A * Vec4<Real>{1, -1, -1, 1};
  //  auto const top_left     = A * Vec4<Real>{-1, 1, -1, 1};
  //  m_plane_base_x = (bottom_right.xyz() / bottom_right.w() - m_bottom_left) /
  //                   (this->plane_width() - 1);
  //  m_plane_base_y = (top_left.xyz() / top_left.w() - m_bottom_left) /
  //                   (this->plane_height() - 1);
  //}
  protected:
   auto set_projection_matrix(mat4 const& P) { m_projection_matrix(P); }
};
//==============================================================================
}  // namespace polymorphic
//==============================================================================
/// \brief Interface for camera implementations.
///
/// Implementations must override the ray method that casts rays through the
/// camera's image plane.
template <floating_point Real, typename Derived>
struct camera_interface : polymorphic::camera<Real> {
  using this_type   = camera_interface<Real, Derived>;
  using parent_type = polymorphic::camera<Real>;
  //----------------------------------------------------------------------------
  using parent_type::parent_type;
  using typename parent_type::mat4;
  //----------------------------------------------------------------------------
  virtual ~camera_interface() = default;
};
//==============================================================================
namespace detail::camera {
//==============================================================================
template <std::floating_point Real>
auto ptr_convertible_to_camera(const volatile polymorphic::camera<Real>*)
    -> std::true_type;
template <typename>
auto ptr_convertible_to_camera(const volatile void*) -> std::false_type;

template <typename>
auto is_derived_from_camera(...) -> std::true_type;
template <typename D>
auto is_derived_from_camera(int)
    -> decltype(ptr_convertible_to_camera(static_cast<D*>(nullptr)));
//==============================================================================
}  // namespace detail::camera
//==============================================================================
template <typename T>
struct is_camera_impl
    : std::integral_constant<
          bool,
          std::is_class_v<T>&& decltype(detail::camera::is_derived_from_camera<
                                        T>(0))::value> {};
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
