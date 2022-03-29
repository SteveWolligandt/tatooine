#ifndef TATOOINE_GL_TEX_TARGET_H
#define TATOOINE_GL_TEX_TARGET_H
//==============================================================================
namespace tatooine::gl::tex {
//==============================================================================
template <std::size_t NumDimenions>
struct target_impl;
template <std::size_t NumDimenions>
struct binding_impl;
//------------------------------------------------------------------------------
template <std::size_t NumDimenions>
static constexpr auto target = target_impl<NumDimenions>::value;
//------------------------------------------------------------------------------
template <std::size_t NumDimenions>
static constexpr auto binding = binding_impl<NumDimenions>::value;
//------------------------------------------------------------------------------
template <>
struct target_impl<1> {
  static constexpr GLenum value   = GL_TEXTURE_1D;
  static constexpr GLenum binding = GL_TEXTURE_BINDING_1D;
};
//------------------------------------------------------------------------------
template <>
struct target_impl<2> {
  static constexpr GLenum value   = GL_TEXTURE_2D;
  static constexpr GLenum binding = GL_TEXTURE_BINDING_2D;
};
//------------------------------------------------------------------------------
template <>
struct target_impl<3> {
  static constexpr GLenum value   = GL_TEXTURE_3D;
  static constexpr GLenum binding = GL_TEXTURE_BINDING_3D;
};
//------------------------------------------------------------------------------
template <>
struct binding_impl<1> {
  static constexpr GLenum value = GL_TEXTURE_BINDING_1D;
};
//------------------------------------------------------------------------------
template <>
struct binding_impl<2> {
  static constexpr GLenum value = GL_TEXTURE_BINDING_2D;
};
//------------------------------------------------------------------------------
template <>
struct binding_impl<3> {
  static constexpr GLenum value = GL_TEXTURE_BINDING_3D;
};
//==============================================================================
}  // namespace tatooine::gl::tex
//==============================================================================
#endif
