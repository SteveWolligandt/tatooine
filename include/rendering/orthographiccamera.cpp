#include <yavin/glincludes.h>
#include <yavin/orthographiccamera.h>
#include <yavin/windowsundefines.h>
//==============================================================================
namespace yavin {
//==============================================================================
orthographiccamera::orthographiccamera(float left, float right, float bottom,
                                       float top, float near, float far,
                                       size_t vp_x, size_t vp_y,
                                       size_t vp_width, size_t vp_height)
    : camera(orthographic_matrix(left, right, bottom, top, near, far), vp_x,
             vp_y, vp_width, vp_height) {}
//------------------------------------------------------------------------------
orthographiccamera::orthographiccamera(float left, float right, float bottom,
                                       float top, float near, float far,
                                       size_t vp_width, size_t vp_height)
    : camera(orthographic_matrix(left, right, bottom, top, near, far), vp_width,
             vp_height) {}
//------------------------------------------------------------------------------
void orthographiccamera::set_projection(float left, float right, float bottom,
                                        float top, float near, float far) {
  m_projection_matrix =
      orthographic_matrix(left, right, bottom, top, near, far);
}
//------------------------------------------------------------------------------
void orthographiccamera::set_projection(float left, float right, float bottom,
                                        float top, float near, float far,
                                        size_t vp_width, size_t vp_height) {
  m_projection_matrix =
      orthographic_matrix(left, right, bottom, top, near, far);
  set_viewport(vp_width, vp_height);
}
//------------------------------------------------------------------------------
void orthographiccamera::set_projection(float left, float right, float bottom,
                                        float top, float near, float far,
                                        size_t vp_x, size_t vp_y,
                                        size_t vp_width, size_t vp_height) {
  m_projection_matrix =
      orthographic_matrix(left, right, bottom, top, near, far);
  set_viewport(vp_x, vp_y, vp_width, vp_height);
}
//==============================================================================
}  // namespace yavin
//==============================================================================
