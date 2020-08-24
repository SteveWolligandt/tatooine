#ifndef YAVIN_CAMERA_H
#define YAVIN_CAMERA_H
//==============================================================================
#include "movable.h"
#include "transform.h"
#include "glfunctions.h"
//==============================================================================
namespace yavin {
//==============================================================================
class camera : public movable {
 public:
   using vec3 = vec<float, 3>;
   using vec4 = vec<float, 4>;
   using mat3 = mat<float, 3, 3>;
   using mat4 = mat<float, 4, 4>;

 protected:
  mat4       m_projection_matrix;
  GLint      m_vp_x, m_vp_y;
  GLsizei    m_vp_w, m_vp_h;

 public:
  camera(const mat4& proj_mat, GLint vp_x, GLint vp_y, GLsizei vp_w, GLsizei vp_h)
      : m_projection_matrix{proj_mat}, m_vp_x{vp_x}, m_vp_y{vp_y}, m_vp_w{vp_w}, m_vp_h{vp_h} {}
  camera(const mat4& proj_mat, GLsizei vp_w, GLsizei vp_h)
      : m_projection_matrix{proj_mat}, m_vp_x{0}, m_vp_y{0}, m_vp_w{vp_w}, m_vp_h{vp_h} {}

  auto&       projection_matrix() { return m_projection_matrix; }
  const auto& projection_matrix() const { return m_projection_matrix; }
  auto        view_matrix() const { return *inverse(m_transform.matrix()); }

  void look_at(const vec3& eye, const vec3& center,
               const vec3& up = {0, 1, 0}) {
    m_transform.look_at(eye, center, up);
  }

  void set_viewport(size_t vp_x, size_t vp_y, size_t vp_w, size_t vp_h) {
    m_vp_x = vp_x;
    m_vp_y = vp_y;
    m_vp_w = vp_w;
    m_vp_h = vp_h;
  }
  void set_viewport(size_t vp_w, size_t vp_h) {
    m_vp_x = 0;
    m_vp_y = 0;
    m_vp_w = vp_w;
    m_vp_h = vp_h;
  }
  auto&       viewport_x() { return m_vp_x; }
  const auto& viewport_x() const { return m_vp_x; }
  auto&       viewport_y() { return m_vp_y; }
  const auto& viewport_y() const { return m_vp_y; }
  auto&       viewport_width() { return m_vp_w; }
  const auto& viewport_width() const { return m_vp_w; }
  auto&       viewport_height() { return m_vp_h; }
  const auto& viewport_height() const { return m_vp_h; }
};

//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
