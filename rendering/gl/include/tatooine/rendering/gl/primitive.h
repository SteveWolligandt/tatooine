#ifndef TATOOINE_RENDERING_GL_PRIMITIVE_H
#define TATOOINE_RENDERING_GL_PRIMITIVE_H
//==============================================================================
#include "glincludes.h"
//==============================================================================
namespace tatooine::rendering::gl {
//==============================================================================
enum Primitive {
  POINTS                   = GL_POINTS,
  LINE_STRIP               = GL_LINE_STRIP,
  LINE_LOOP                = GL_LINE_LOOP,
  LINES                    = GL_LINES,
  LINE_STRIP_ADJACENCY     = GL_LINE_STRIP_ADJACENCY,
  LINES_ADJACENCY          = GL_LINES_ADJACENCY,
  TRIANGLE_STRIP           = GL_TRIANGLE_STRIP,
  TRIANGLE_FAN             = GL_TRIANGLE_FAN,
  TRIANGLES                = GL_TRIANGLES,
  TRIANGLE_STRIP_ADJACENCY = GL_TRIANGLE_STRIP_ADJACENCY,
  TRIANGLES_ADJACENCY      = GL_TRIANGLES_ADJACENCY,
  PATCHES                  = GL_PATCHES
};
//==============================================================================
}  // namespace tatooine::rendering::gl
//==============================================================================
#endif
