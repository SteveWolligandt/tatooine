#include <yavin/glincludes.h>
#include <yavin/perspectivecamera.h>
#include <yavin/windowsundefines.h>
//==============================================================================
namespace yavin {
//==============================================================================
perspectivecamera::perspectivecamera(float fovy, float aspect, float near,
                                     float far, GLint vp_x, GLint vp_y,
                                     GLsizei vp_width, GLsizei vp_height)
    : camera{perspective_matrix<float>(fovy, aspect, near, far), vp_x, vp_y,
             vp_width, vp_height},
      m_fovy{fovy},
      m_aspect{aspect},
      m_near{near},
      m_far{far} {}
//------------------------------------------------------------------------------
perspectivecamera::perspectivecamera(float fovy, float aspect, float near,
                                     float far, GLsizei vp_width,
                                     GLsizei vp_height)
    : camera{perspective_matrix<float>(fovy, aspect, near, far), vp_width,
             vp_height},
      m_fovy{fovy},
      m_aspect{aspect},
      m_near{near},
      m_far{far} {}
//------------------------------------------------------------------------------
void perspectivecamera::set_projection(float fovy, float aspect, float near,
                                       float far) {
  m_projection_matrix = perspective_matrix<float>(fovy, aspect, near, far);
  m_fovy              = fovy;
  m_aspect            = aspect;
  m_near              = near;
  m_far               = far;
}
//------------------------------------------------------------------------------
void perspectivecamera::set_projection(float fovy, float aspect, float near,
                                       float far, GLsizei vp_width,
                                       GLsizei vp_height) {
  set_projection(fovy, aspect, near, far);
  set_viewport(vp_width, vp_height);
  m_fovy   = fovy;
  m_aspect = aspect;
  m_near   = near;
  m_far    = far;
}
//------------------------------------------------------------------------------
void perspectivecamera::set_projection(float fovy, float aspect, float near,
                                       float far, GLint vp_x, GLint vp_y,
                                       GLsizei vp_width, GLsizei vp_height) {
  set_projection(fovy, aspect, near, far);
  set_viewport(vp_x, vp_y, vp_width, vp_height);
  m_fovy   = fovy;
  m_aspect = aspect;
  m_near   = near;
  m_far    = far;
}
//==============================================================================
}  // namespace yavin
//==============================================================================
