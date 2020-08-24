#ifndef __YAVIN_ORTHOGONAL_CAMERA_H__
#define __YAVIN_ORTHOGONAL_CAMERA_H__

#include "camera.h"
#include "dllexport.h"

//==============================================================================
namespace yavin {
//==============================================================================

class orthographiccamera : public camera {
 public:
  DLL_API orthographiccamera(float left, float right,
                             float bottom, float top,
                             float near, float far,
                             size_t vp_x, size_t vp_y,
                             size_t vp_width, size_t vp_height);
  DLL_API orthographiccamera(float left, float right,
                             float bottom, float top,
                             float near, float far,
                             size_t vp_width, size_t vp_height);
  DLL_API void set_projection(float left, float right,
                              float bottom, float top,
                              float near, float far);
  DLL_API void set_projection(float left, float right,
                              float bottom, float top,
                              float near, float far, size_t vp_width,
                              size_t vp_height);
  DLL_API void set_projection(float left, float right,
                              float bottom, float top,
                              float near, float far,
                              size_t vp_x, size_t vp_y,
                              size_t vp_width, size_t vp_height);
};
//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
