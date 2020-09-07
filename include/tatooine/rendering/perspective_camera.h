#ifndef TATOOINE_RENDERING_PERSPECTIVE_CAMERA_H
#define TATOOINE_RENDERING_PERSPECTIVE_CAMERA_H
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
template <real_number Real>
class perspective_camera : public camera<Real> {
 public:
  using real_t   = Real;
  using parent_t = camera<real_t>;
  using this_t   = perspective_camera<real_t>;
  using vec3     = vec<real_t, 3>;
  using mat4     = mat<real_t, 4, 4>;

 private:
  //----------------------------------------------------------------------------
  // class variables
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // member variables
  //----------------------------------------------------------------------------
  vec3   m_eye, m_lookat, m_up;
  vec3   m_bottom_left;
  vec3   m_plane_base_x, m_plane_base_y;
  real_t m_fov;

 public:
  //----------------------------------------------------------------------------
  // constructors / destructor
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  perspective_camera(vec3 const& eye, vec3 const& lookat, vec3 const& up,
                     real_t const fov, size_t const res_x, size_t const res_y)
      : parent_t{res_x, res_y},
        m_eye{eye},
        m_lookat{lookat},
        m_up{up},
        m_fov{fov} {
    setup();
  }
  //----------------------------------------------------------------------------
  /// Constructor generates bottom left image plane pixel position and pixel
  /// offset size.
  perspective_camera(vec3 const& eye, vec3 const& lookat, real_t fov,
                     size_t res_x, size_t res_y)
      : perspective_camera(eye, lookat, vec3{0, 1, 0}, fov, res_x, res_y) {}
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
  //============================================================================
 private:
  void setup() {
    vec3 const   view_dir = normalize(m_lookat - m_eye);
    vec3 const   u        = cross(m_up, view_dir);
    vec3 const   v        = cross(view_dir, u);
    real_t const plane_half_width =
        std::tan(m_fov / real_t(2) * real_t(M_PI) / real_t(180));
    real_t const plane_half_height = plane_half_width * this->aspect_ratio();
    m_bottom_left =
        m_eye + view_dir - u * plane_half_width - v * plane_half_height;
    m_plane_base_x = u * 2 * plane_half_width / (this->plane_width() - 1);
    m_plane_base_y = v * 2 * plane_half_height / (this->plane_height() - 1);
  }
  //----------------------------------------------------------------------------
 public:
  void set_eye(vec3 const& eye) {
    m_eye = eye;
    setup();
  }
  //------------------------------------------------------------------------------
  void set_resolution(size_t const res_x, size_t const res_y) {
    parent_t::set_resolution(res_x, res_y);
    setup();
  }
  //------------------------------------------------------------------------------
  void set_lookat(vec3 const lookat) {
    m_lookat = lookat;
    setup();
  }
  //------------------------------------------------------------------------------
  void set_up(vec3 const up) {
    m_up = up;
    setup();
  }
  //------------------------------------------------------------------------------
  void set_fov(real_t const fov) {
    m_fov = fov;
    setup();
  }
  //----------------------------------------------------------------------------
  void look_at(vec3 const& eye, vec3 const& lookat,
               vec3 const& up = {0, 1, 0}) {
    m_eye    = eye;
    m_lookat = lookat;
    m_up     = up;
    setup();
  }
  //------------------------------------------------------------------------------
  void look_at(vec3 const& eye, vec3 const& lookat, vec3 const& up,
               real_t fov) {
    m_eye    = eye;
    m_lookat = lookat;
    m_up     = up;
    m_fov    = fov;
    setup();
  }
  //----------------------------------------------------------------------------
  void setup(vec3 const& eye, vec3 const& lookat, vec3 const& up, real_t fov,
             size_t res_x, size_t res_y) {
    this->set_resolution(res_x, res_y);
    m_eye    = eye;
    m_lookat = lookat;
    m_up     = up;
    m_fov    = fov;
    setup();
  }
  //----------------------------------------------------------------------------
  std::unique_ptr<parent_t> clone() const override {
    return std::unique_ptr<this_t>(new this_t{*this});
  }
  //----------------------------------------------------------------------------
  auto projection_matrix(real_t const near, real_t const far) const
      -> mat4 {
    real_t const plane_half_width =
        std::tan(m_fov / real_t(2) * real_t(M_PI) / real_t(180)) * near;
    real_t const r = this->aspect_ratio() * plane_half_width;
    real_t const l = -r;
    real_t const t = plane_half_width;
    real_t const b = -t;
    return {{2 * near / (r - l), real_t(0), (r + l) / (r - l), real_t(0)},
            {real_t(0), 2 * near / (t - b), (t + b) / (t - b), real_t(0)},
            {real_t(0), real_t(0), -(far + near) / (far - near),
             -2 * far * near / (far - near)},
            {real_t(0), real_t(0), real_t(-1), real_t(0)}};
  }
  //----------------------------------------------------------------------------
  auto transform_matrix(real_t const near, real_t const far) const
      -> mat4 {
    return look_at_matrix(m_eye, m_lookat, m_up);
  }
  //----------------------------------------------------------------------------
  auto view_matrix(real_t const near, real_t const far) const -> mat4 {
    return inv(transform_matrix(near, far));
  }
};
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
