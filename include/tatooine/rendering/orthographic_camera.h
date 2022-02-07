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
template <arithmetic Real>
class orthographic_camera
    : public camera_interface<Real, orthographic_camera<Real>> {
 public:
  using real_t      = Real;
  using this_t      = orthographic_camera<Real>;
  using parent_type = camera_interface<Real, this_t>;
  using vec3        = vec<Real, 3>;
  using mat4        = mat<Real, 4, 4>;

  using parent_type::d;
  using parent_type::depth;
  using parent_type::f;
  using parent_type::far;
  using parent_type::n;
  using parent_type::near;
  using parent_type::setup;

 private:
  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
  Real m_left;
  Real m_right;
  Real m_bottom;
  Real m_top;

 public:
  //----------------------------------------------------------------------------
  // getter / setter
  //----------------------------------------------------------------------------
  auto constexpr left() const { return m_left; }
  //----------------------------------------------------------------------------
  auto constexpr l() const { return left(); }
  //----------------------------------------------------------------------------
  auto constexpr right() const { return m_right; }
  //----------------------------------------------------------------------------
  auto constexpr r() const { return right(); }
  //----------------------------------------------------------------------------
  auto constexpr bottom() const { return m_bottom; }
  //----------------------------------------------------------------------------
  auto constexpr b() const { return bottom(); }
  //----------------------------------------------------------------------------
  auto constexpr top() const { return m_top; }
  //----------------------------------------------------------------------------
  auto constexpr t() const { return top(); }
  //----------------------------------------------------------------------------
  auto constexpr width() const { return right() - left(); }
  //----------------------------------------------------------------------------
  auto constexpr w() const { return width(); }
  //----------------------------------------------------------------------------
  auto constexpr height() const { return top() - bottom(); }
  //----------------------------------------------------------------------------
  auto constexpr h() const { return height(); }
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                vec3 const& up, Real const left,
                                Real const right, Real const bottom,
                                Real const top, Real const near, Real const far,
                                Vec4<std::size_t> const & viewport)
      : parent_type{eye,  lookat, up,
                    near, far,    viewport},
        m_left{left},
        m_right{right},
        m_bottom{bottom},
        m_top{top} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                vec3 const& up, Real const left,
                                Real const right, Real const bottom,
                                Real const top, Real const near, Real const far,
                                std::size_t const res_x,
                                std::size_t const res_y)
      : parent_type{eye,  lookat, up,
                    near, far,    Vec4<std::size_t>{0, 0, res_x, res_y}},
        m_left{left},
        m_right{right},
        m_bottom{bottom},
        m_top{top} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                Real const left, Real const right,
                                Real const bottom, Real const top,
                                Real const near, Real const far,
                                Vec4<std::size_t> const& viewport)
      : parent_type{eye,  lookat, vec3{0, 1, 0},
                    near, far,    viewport},
        m_left{left},
        m_right{right},
        m_bottom{bottom},
        m_top{top} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                Real const left, Real const right,
                                Real const bottom, Real const top,
                                Real const near, Real const far,
                                std::size_t const res_x,
                                std::size_t const res_y)
      : parent_type{eye,  lookat, vec3{0, 1, 0},
                    near, far,    Vec4<std::size_t>{0, 0, res_x, res_y}},
        m_left{left},
        m_right{right},
        m_bottom{bottom},
        m_top{top} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                vec3 const& up, Real const height,
                                Real const near, Real const far,
                                Vec4<std::size_t> const& viewport)
      : orthographic_camera{eye,
                            lookat,
                            up,
                            -height / 2 * Real(viewport[2]) * Real(viewport[3]),
                            height / 2 * Real(viewport[2]) * Real(viewport[3]),
                            -height / 2,
                            height / 2,
                            near,
                            far,
                            viewport} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  constexpr orthographic_camera(vec3 const& eye, vec3 const& lookat,
                                vec3 const& up, Real const height,
                                Real const near, Real const far,
                                std::size_t const res_x,
                                std::size_t const res_y)
      : orthographic_camera{eye,
                            lookat,
                            up,
                            -height / 2 * Real(res_x) * Real(res_y),
                            height / 2 * Real(res_x) * Real(res_y),
                            -height / 2,
                            height / 2,
                            near,
                            far,
                            res_x,
                            res_y} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, Real const height,
                      Real const near, Real const far,
                      Vec4<std::size_t> const& viewport)
      : orthographic_camera{eye,  lookat, vec3{0, 1, 0}, height,
                            near, far,    viewport} {}
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, Real const height,
                      Real const near, Real const far, std::size_t const res_x,
                      std::size_t const res_y)
      : orthographic_camera{eye,
                            lookat,
                            vec3{0, 1, 0},
                            height,
                            near,
                            far,
                            Vec4<std::size_t>{0, 0, res_x, res_y}} {}
  //----------------------------------------------------------------------------
  ~orthographic_camera() = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
  auto constexpr projection_matrix() const -> mat4 {
    return orthographic_matrix(l(), r(), b(), t(), n(), f());
  }
  //----------------------------------------------------------------------------
  auto constexpr setup(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                       Real const width, Real const height, Real const near,
                       Real const far, std::size_t const res_x,
                       std::size_t const res_y) -> void {
    this->set_eye_without_update(eye);
    this->set_lookat_without_update(lookat);
    this->set_up_without_update(up);
    this->set_resolution_without_update(res_x, res_y);
    m_top    = height / 2;
    m_bottom = -m_top;
    m_right  = width / 2;
    m_left   = -m_right;
    this->set_near_without_update(near);
    this->set_far_without_update(far);
    setup();
  }
  //----------------------------------------------------------------------------
  auto constexpr setup(Real const width, Real const height) -> void {
    m_top    = height / 2;
    m_bottom = -m_top;
    m_right  = width / 2;
    m_left   = -m_right;
    setup();
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
