#ifndef YAVIN_TEX_COMPONENTS_H
#define YAVIN_TEX_COMPONENTS_H
//==============================================================================
#include "glincludes.h"
//==============================================================================
namespace yavin {
//==============================================================================
struct R {
  static constexpr std::size_t num_components = 1;
};
struct RG {
  static constexpr std::size_t num_components = 2;
};
struct RGB {
  static constexpr std::size_t num_components = 3;
};
struct RGBA {
  static constexpr std::size_t num_components = 4;
};
struct BGR {
  static constexpr std::size_t num_components = 3;
};
struct BGRA {
  static constexpr std::size_t num_components = 4;
};
struct Depth {
  static constexpr std::size_t num_components = 1;
};
//==============================================================================
template <typename C> struct is_depth_component : std::false_type {};
template <> struct is_depth_component<Depth> : std::true_type {};
//------------------------------------------------------------------------------
template <typename C> struct is_color_component : std::false_type {};
template <> struct is_color_component<R> : std::true_type {};
template <> struct is_color_component<RG> : std::true_type {};
template <> struct is_color_component<RGB> : std::true_type {};
template <> struct is_color_component<RGBA> : std::true_type {};
template <> struct is_color_component<BGR> : std::true_type {};
template <> struct is_color_component<BGRA> : std::true_type {};
//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
