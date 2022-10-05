#ifndef TATOOINE_GL_TEX_COMPONENTS_H
#define TATOOINE_GL_TEX_COMPONENTS_H
//==============================================================================
#include <tatooine/gl/glincludes.h>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
struct R {
  static constexpr GLsizei num_components = 1;
};
struct RG {
  static constexpr GLsizei num_components = 2;
};
struct RGB {
  static constexpr GLsizei num_components = 3;
};
struct RGBA {
  static constexpr GLsizei num_components = 4;
};
struct BGR {
  static constexpr GLsizei num_components = 3;
};
struct BGRA {
  static constexpr GLsizei num_components = 4;
};
struct Depth {
  static constexpr GLsizei num_components = 1;
};
//==============================================================================
template <typename T>
static auto constexpr texture_depth_component = either_of<T, Depth>;
//------------------------------------------------------------------------------
template <typename T>
static auto constexpr texture_color_component =
    either_of<T, R, RG, RGB, RGBA, BGR, BGRA>;
//==============================================================================
template <typename T>
concept texture_component =
    texture_color_component<T> || texture_depth_component<T>;
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
