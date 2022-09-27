#ifndef TATOOINE_GL_BUFFER_USAGE_H
#define TATOOINE_GL_BUFFER_USAGE_H
//==============================================================================
#include <tatooine/gl/glincludes.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
enum class buffer_usage : GLenum {
  STREAM_DRAW  = GL_STREAM_DRAW,
  STREAM_READ  = GL_STREAM_READ,
  STREAM_COPY  = GL_STREAM_COPY,
  STATIC_DRAW  = GL_STATIC_DRAW,
  STATIC_READ  = GL_STATIC_READ,
  STATIC_COPY  = GL_STATIC_COPY,
  DYNAMIC_DRAW = GL_DYNAMIC_DRAW,
  DYNAMIC_READ = GL_DYNAMIC_READ,
  DYNAMIC_COPY = GL_DYNAMIC_COPY
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
