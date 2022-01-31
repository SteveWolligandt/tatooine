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
class orthographic_camera : public camera_interface<Real, orthographic_camera<Real>> {
 public:
  using real_t   = Real;
  using this_t   = orthographic_camera<Real>;
  using parent_t = camera_interface<Real, this_t>;
  using vec3     = vec<Real, 3>;
  using mat4     = mat<Real, 4, 4>;

  using parent_t::d;
  using parent_t::depth;
  using parent_t::f;
  using parent_t::far;
  using parent_t::n;
  using parent_t::near;
  using parent_t::setup;

 private:
  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
  vec3 m_bottom_left;
  vec3 m_plane_base_x, m_plane_base_y;
  Real m_left;
  Real m_right;
  Real m_bottom;
  Real m_top;

 public:
  //----------------------------------------------------------------------------
  // getter / setter
  //----------------------------------------------------------------------------
  auto left() const { return m_left; }
  //----------------------------------------------------------------------------
  auto l() const { return left(); }
  //----------------------------------------------------------------------------
  auto right() const { return m_right; }
  //----------------------------------------------------------------------------
  auto r() const { return right(); }
  //----------------------------------------------------------------------------
  auto bottom() const { return m_bottom; }
  //----------------------------------------------------------------------------
  auto b() const { return bottom(); }
  //----------------------------------------------------------------------------
  auto top() const { return m_top; }
  //----------------------------------------------------------------------------
  auto t() const { return top(); }
  //----------------------------------------------------------------------------
  auto width() const { return right() - left(); }
  //----------------------------------------------------------------------------
  auto w() const { return width(); }
  //----------------------------------------------------------------------------
  auto height() const { return top() - bottom(); }
  //----------------------------------------------------------------------------
  auto h() const { return height(); }
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                      Real const left, Real const right, Real const bottom,
                      Real const top, Real const near, Real const far,
                      std::size_t const res_x, std::size_t const res_y)
      : parent_t{eye, lookat, up, near, far, res_x, res_y},
        m_left{left},
        m_right{right},
        m_bottom{bottom},
        m_top{top} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, Real const left,
                      Real const right, Real const bottom, Real const top,
                      Real const near, Real const far, std::size_t const res_x,
                      std::size_t const res_y)
      : parent_t{eye, lookat, vec3{0, 1, 0}, near, far, res_x, res_y},
        m_left{left},
        m_right{right},
        m_bottom{bottom},
        m_top{top} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                      Real const height, Real const near, Real const far,
                      std::size_t const res_x, std::size_t const res_y)
      : orthographic_camera{eye,        lookat, up,  -1,    1,    -height / 2,
                            height / 2, near,   far, res_x, res_y} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  orthographic_camera(vec3 const& eye, vec3 const& lookat, Real const height,
                      Real const near, Real const far, std::size_t const res_x,
                      std::size_t const res_y)
      : orthographic_camera{eye,  lookat, vec3{0, 1, 0}, height,
                            near, far,    res_x,         res_y} {}
  //----------------------------------------------------------------------------
  ~orthographic_camera() = default;
  //----------------------------------------------------------------------------
  // object methods
  //----------------------------------------------------------------------------
 public:
  //----------------------------------------------------------------------------
  auto projection_matrix() const -> mat4 {
    static auto constexpr z = Real(0);
    static auto constexpr o = Real(1);

    auto const iw = 1 / w();
    auto const ih = 1 / h();
    auto const id = 1 / d();
    auto const xs = 2 * iw;
    auto const xt = (r() + l()) * iw;
    auto const ys = 2 * ih;
    auto const yt = (b() + t()) * ih;
    auto const zs = -2 * id;
    auto const zt = -(f() + n()) * id;
    return mat4{{xs, z, z, xt}, {z, ys, z, yt}, {z, z, zs, zt}, {z, z, z, o}};
  }
  //----------------------------------------------------------------------------
  auto setup(vec3 const& eye, vec3 const& lookat, vec3 const& up,
             Real const height, Real const near, Real const far,
             std::size_t const res_x, std::size_t const res_y) -> void {
    this->set_eye(eye);
    this->set_lookat(lookat);
    this->set_up(up);
    this->set_resolution(res_x, res_y);
    m_top    = height / 2;
    m_bottom = -top;
    m_left   = -1;
    m_right  = 1;
    this->set_near(near);
    this->set_far(far);
    setup();
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
