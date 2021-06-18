#ifndef YAVIN_TEX_TARGET_H
#define YAVIN_TEX_TARGET_H
//==============================================================================
namespace yavin::tex {
//==============================================================================
template <unsigned int n>
struct target;
//------------------------------------------------------------------------------
template <unsigned int n>
static constexpr auto target_v = target<n>::value;
//------------------------------------------------------------------------------
template <unsigned int n>
static constexpr auto target_binding = target<n>::binding;
//------------------------------------------------------------------------------
template <>
struct target<1> {
  static constexpr GLenum value   = GL_TEXTURE_1D;
  static constexpr GLenum binding = GL_TEXTURE_BINDING_1D;
};
//------------------------------------------------------------------------------
template <>
struct target<2> {
  static constexpr GLenum value   = GL_TEXTURE_2D;
  static constexpr GLenum binding = GL_TEXTURE_BINDING_2D;
};
//------------------------------------------------------------------------------
template <>
struct target<3> {
  static constexpr GLenum value   = GL_TEXTURE_3D;
  static constexpr GLenum binding = GL_TEXTURE_BINDING_3D;
};
//==============================================================================
}  // namespace yavin::tex
//==============================================================================

#endif
