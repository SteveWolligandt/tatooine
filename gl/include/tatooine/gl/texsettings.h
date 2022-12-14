#ifndef TATOOINE_GL_TEXSETTINGS_H
#define TATOOINE_GL_TEXSETTINGS_H
//==============================================================================
#include "glincludes.h"
#include "gltype.h"
#include "texcomponents.h"
//==============================================================================
namespace tatooine::gl::tex {
//==============================================================================
template <typename T, typename format>
struct settings;
//------------------------------------------------------------------------------
// GLubyte
//------------------------------------------------------------------------------
template <>
struct settings<GLubyte, R> {
  using real_type                         = GLubyte;
  using components_type                            = R;
  static constexpr GLint  internal_format = GL_R8UI;
  static constexpr GLenum format          = GL_RED;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLubyte, RG> {
  using real_type                         = GLubyte;
  using components_type                            = RG;
  static constexpr GLint  internal_format = GL_RG8UI;
  static constexpr GLenum format          = GL_RG;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLubyte, RGB> {
  using real_type                         = GLubyte;
  using components_type                            = RGB;
  static constexpr GLint  internal_format = GL_RGB8UI;
  static constexpr GLenum format          = GL_RGB;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLubyte, RGBA> {
  using real_type                         = GLubyte;
  using components_type                            = RGBA;
  static constexpr GLint  internal_format = GL_RGBA;
  static constexpr GLenum format          = GL_RGBA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLubyte, BGR> {
  using real_type                         = GLubyte;
  using components_type                            = BGR;
  static constexpr GLint  internal_format = GL_RGB8UI;
  static constexpr GLenum format          = GL_BGR;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLubyte, BGRA> {
  using real_type                         = GLubyte;
  using components_type                            = BGRA;
  static constexpr GLint  internal_format = GL_RGBA8UI;
  static constexpr GLenum format          = GL_BGRA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
//------------------------------------------------------------------------------
// GLbyte
//------------------------------------------------------------------------------
template <>
struct settings<GLbyte, R> {
  using real_type                         = GLbyte;
  using components_type                            = R;
  static constexpr GLint  internal_format = GL_R8I;
  static constexpr GLenum format          = GL_RED;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLbyte, RG> {
  using real_type                         = GLbyte;
  using components_type                            = RG;
  static constexpr GLint  internal_format = GL_RG8I;
  static constexpr GLenum format          = GL_RG;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLbyte, RGB> {
  using real_type                         = GLbyte;
  using components_type                            = RGB;
  static constexpr GLint  internal_format = GL_RGB8I;
  static constexpr GLenum format          = GL_RGB;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLbyte, RGBA> {
  using real_type                         = GLbyte;
  using components_type                            = RGBA;
  static constexpr GLint  internal_format = GL_RGBA8I;
  static constexpr GLenum format          = GL_RGBA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLbyte, BGR> {
  using real_type                         = GLbyte;
  using components_type                            = BGR;
  static constexpr GLint  internal_format = GL_RGB8I;
  static constexpr GLenum format          = GL_BGR;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLbyte, BGRA> {
  using real_type                         = GLbyte;
  using components_type                            = BGRA;
  static constexpr GLint  internal_format = GL_RGBA8I;
  static constexpr GLenum format          = GL_BGRA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
//------------------------------------------------------------------------------
// GLushort
template <>
struct settings<GLushort, R> {
  using real_type                         = GLushort;
  using components_type                            = R;
  static constexpr GLint  internal_format = GL_R16UI;
  static constexpr GLenum format          = GL_RED;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLushort, RG> {
  using real_type                         = GLushort;
  using components_type                            = RG;
  static constexpr GLint  internal_format = GL_RG16UI;
  static constexpr GLenum format          = GL_RG;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLushort, RGB> {
  using real_type                         = GLushort;
  using components_type                            = RGB;
  static constexpr GLint  internal_format = GL_RGB16UI;
  static constexpr GLenum format          = GL_RGB;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLushort, RGBA> {
  using real_type                         = GLushort;
  using components_type                            = RGBA;
  static constexpr GLint  internal_format = GL_RGBA16UI;
  static constexpr GLenum format          = GL_RGBA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLushort, BGR> {
  using real_type                         = GLushort;
  using components_type                            = BGR;
  static constexpr GLint  internal_format = GL_RGB16UI;
  static constexpr GLenum format          = GL_BGR;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLushort, BGRA> {
  using real_type                         = GLushort;
  using components_type                            = BGRA;
  static constexpr GLint  internal_format = GL_RGBA16UI;
  static constexpr GLenum format          = GL_BGRA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
//------------------------------------------------------------------------------
// GLshort
//------------------------------------------------------------------------------
template <>
struct settings<GLshort, R> {
  using real_type                         = GLshort;
  using components_type                            = R;
  static constexpr GLint  internal_format = GL_R16I;
  static constexpr GLenum format          = GL_RED;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLshort, RG> {
  using real_type                         = GLshort;
  using components_type                            = RG;
  static constexpr GLint  internal_format = GL_RG16I;
  static constexpr GLenum format          = GL_RG;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLshort, RGB> {
  using real_type                         = GLshort;
  using components_type                            = RGB;
  static constexpr GLint  internal_format = GL_RGB16I;
  static constexpr GLenum format          = GL_RGB;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLshort, RGBA> {
  using real_type                         = GLshort;
  using components_type                            = RGBA;
  static constexpr GLint  internal_format = GL_RGBA16I;
  static constexpr GLenum format          = GL_RGBA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLshort, BGR> {
  using real_type                         = GLshort;
  using components_type                            = BGR;
  static constexpr GLint  internal_format = GL_RGB16I;
  static constexpr GLenum format          = GL_BGR;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLshort, BGRA> {
  using real_type                         = GLshort;
  using components_type                            = BGRA;
  static constexpr GLint  internal_format = GL_RGBA16I;
  static constexpr GLenum format          = GL_BGRA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
//------------------------------------------------------------------------------
// GLuint
//------------------------------------------------------------------------------
template <>
struct settings<GLuint, R> {
  using real_type                         = GLuint;
  using components_type                            = R;
  static constexpr GLint  internal_format = GL_R32UI;
  static constexpr GLenum format          = GL_RED_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLuint, RG> {
  using real_type                         = GLuint;
  using components_type                            = RG;
  static constexpr GLint  internal_format = GL_RG32UI;
  static constexpr GLenum format          = GL_RG_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLuint, RGB> {
  using real_type                         = GLuint;
  using components_type                            = RGB;
  static constexpr GLint  internal_format = GL_RGB32UI;
  static constexpr GLenum format          = GL_RGB_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLuint, RGBA> {
  using real_type                         = GLuint;
  using components_type                            = RGBA;
  static constexpr GLint  internal_format = GL_RGBA32UI;
  static constexpr GLenum format          = GL_RGBA_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLuint, BGR> {
  using real_type                         = GLuint;
  using components_type                            = BGR;
  static constexpr GLint  internal_format = GL_RGB32UI;
  static constexpr GLenum format          = GL_BGR_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLuint, BGRA> {
  using real_type                         = GLuint;
  using components_type                            = BGRA;
  static constexpr GLint  internal_format = GL_RGBA32UI;
  static constexpr GLenum format          = GL_BGRA_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
//------------------------------------------------------------------------------
// GLint
//------------------------------------------------------------------------------
template <>
struct settings<GLint, R> {
  using real_type                         = GLint;
  using components_type                            = R;
  static constexpr GLint  internal_format = GL_R32I;
  static constexpr GLenum format          = GL_RED_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLint, RG> {
  using real_type                         = GLint;
  using components_type                            = RG;
  static constexpr GLint  internal_format = GL_RG32I;
  static constexpr GLenum format          = GL_RG_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLint, RGB> {
  using real_type                         = GLint;
  using components_type                            = RGB;
  static constexpr GLint  internal_format = GL_RGB32I;
  static constexpr GLenum format          = GL_RGB_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLint, RGBA> {
  using real_type                         = GLint;
  using components_type                            = RGBA;
  static constexpr GLint  internal_format = GL_RGBA32I;
  static constexpr GLenum format          = GL_RGBA_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLint, BGR> {
  using real_type                         = GLint;
  using components_type                            = BGR;
  static constexpr GLint  internal_format = GL_RGB32I;
  static constexpr GLenum format          = GL_BGR_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLint, BGRA> {
  using real_type                         = GLint;
  using components_type                            = BGRA;
  static constexpr GLint  internal_format = GL_RGBA32I;
  static constexpr GLenum format          = GL_BGRA_INTEGER;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
//------------------------------------------------------------------------------
// GLhalf
//------------------------------------------------------------------------------
template <>
struct settings<gl_half, R> {
  using real_type                         = GLhalf;
  using components_type                            = R;
  static constexpr GLint  internal_format = GL_R16F;
  static constexpr GLenum format          = GL_RED;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<gl_half, RG> {
  using real_type                         = GLhalf;
  using components_type                            = RG;
  static constexpr GLint  internal_format = GL_RG16F;
  static constexpr GLenum format          = GL_RG;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<gl_half, RGB> {
  using real_type                         = GLhalf;
  using components_type                            = RGB;
  static constexpr GLint  internal_format = GL_RGB16F;
  static constexpr GLenum format          = GL_RGB;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<gl_half, RGBA> {
  using real_type                         = GLhalf;
  using components_type                            = RGBA;
  static constexpr GLint  internal_format = GL_RGBA16F;
  static constexpr GLenum format          = GL_RGBA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<gl_half, BGR> {
  using real_type                         = GLhalf;
  using components_type                            = BGR;
  static constexpr GLint  internal_format = GL_RGB16F;
  static constexpr GLenum format          = GL_BGR;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<gl_half, BGRA> {
  using real_type                         = GLhalf;
  using components_type                            = BGRA;
  static constexpr GLint  internal_format = GL_RGBA16F;
  static constexpr GLenum format          = GL_BGRA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
//------------------------------------------------------------------------------
// GLfloat
template <>
struct settings<GLfloat, R> {
  using real_type                         = GLfloat;
  using components_type                            = R;
  static constexpr GLint  internal_format = GL_R32F;
  static constexpr GLenum format          = GL_RED;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLfloat, RG> {
  using real_type                         = GLfloat;
  using components_type                            = RG;
  static constexpr GLint  internal_format = GL_RG32F;
  static constexpr GLenum format          = GL_RG;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLfloat, RGB> {
  using real_type                         = GLfloat;
  using components_type                            = RGB;
  static constexpr GLint  internal_format = GL_RGB32F;
  static constexpr GLenum format          = GL_RGB;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLfloat, RGBA> {
  using real_type                         = GLfloat;
  using components_type                            = RGBA;
  static constexpr GLint  internal_format = GL_RGBA32F;
  static constexpr GLenum format          = GL_RGBA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLfloat, BGR> {
  using real_type                         = GLfloat;
  using components_type                            = BGR;
  static constexpr GLint  internal_format = GL_RGB32F;
  static constexpr GLenum format          = GL_BGR;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLfloat, BGRA> {
  using real_type                         = GLfloat;
  using components_type                            = BGRA;
  static constexpr GLint  internal_format = GL_RGBA32F;
  static constexpr GLenum format          = GL_BGRA;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
//------------------------------------------------------------------------------
// Depth
template <>
struct settings<GLushort, Depth> {
  using real_type                         = GLushort;
  using components_type                            = Depth;
  static constexpr GLint  internal_format = GL_DEPTH_COMPONENT16;
  static constexpr GLenum format          = GL_DEPTH_COMPONENT;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLuint, Depth> {
  using real_type                         = GLint;
  using components_type                            = Depth;
  static constexpr GLint  internal_format = GL_DEPTH_COMPONENT32;
  static constexpr GLenum format          = GL_DEPTH_COMPONENT;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
struct depth24 {};
template <>
struct settings<depth24, Depth> {
  using real_type                         = depth24;
  using components_type                            = Depth;
  static constexpr GLint  internal_format = GL_DEPTH_COMPONENT24;
  static constexpr GLenum format          = GL_DEPTH_COMPONENT;
  static constexpr GLenum type            = GL_UNSIGNED_BYTE;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct settings<GLfloat, Depth> {
  using real_type                         = GLint;
  using components_type                            = Depth;
  static constexpr GLint  internal_format = GL_DEPTH_COMPONENT32F;
  static constexpr GLenum format          = GL_DEPTH_COMPONENT;
  static constexpr GLenum type            = gl_type_v<real_type>;
};
//==============================================================================
}  // namespace tatooine::gl::tex
//==============================================================================
#endif
