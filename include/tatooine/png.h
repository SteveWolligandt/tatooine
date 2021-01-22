#ifndef TATOOINE_PNG_H
#define TATOOINE_PNG_H
//==============================================================================
#ifdef TATOOINE_HAS_PNG_SUPPORT
#include <png++/png.hpp>
#endif
//==============================================================================
namespace tatooine {
//==============================================================================
static constexpr auto has_png_support() {
#ifdef TATOOINE_HAS_PNG_SUPPORT
  return true;
#else
  return false;
#endif
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
