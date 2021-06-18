#ifndef YAVIN_GLTYPE_H
#define YAVIN_GLTYPE_H
//==============================================================================
#include "glincludes.h"
//==============================================================================
namespace yavin {
//==============================================================================
struct gl_half;

template <typename T>
struct gl_type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto gl_type_v = gl_type<T>::value;
//------------------------------------------------------------------------------
//template <>
//struct gl_type<GLfixed> {
//  static constexpr GLenum value = GL_FIXED;
//};
//------------------------------------------------------------------------------
template <>
struct gl_type<gl_half> {
  static constexpr GLenum value = GL_HALF_FLOAT;
};
//------------------------------------------------------------------------------
template <>
struct gl_type<GLfloat> {
  static constexpr GLenum value = GL_FLOAT;
};
//------------------------------------------------------------------------------
template <>
struct gl_type<GLdouble> {
  static constexpr GLenum value = GL_DOUBLE;
};
//------------------------------------------------------------------------------
template <>
struct gl_type<GLubyte> {
  static constexpr GLenum value = GL_UNSIGNED_BYTE;
};
//------------------------------------------------------------------------------
template <>
struct gl_type<GLbyte> {
  static constexpr GLenum value = GL_BYTE;
};
//------------------------------------------------------------------------------
template <>
struct gl_type<GLushort> {
  static constexpr GLenum value = GL_UNSIGNED_SHORT;
};
//------------------------------------------------------------------------------
template <>
struct gl_type<GLshort> {
  static constexpr GLenum value = GL_SHORT;
};
//------------------------------------------------------------------------------
template <>
struct gl_type<GLuint> {
  static constexpr GLenum value = GL_UNSIGNED_INT;
};
//------------------------------------------------------------------------------
template <>
struct gl_type<GLint> {
  static constexpr GLenum value = GL_INT;
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
