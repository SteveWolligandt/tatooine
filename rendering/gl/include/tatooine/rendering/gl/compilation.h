#ifndef __YAVIN_COMPILATION_H__
#define __YAVIN_COMPILATION_H__

//==============================================================================
namespace yavin {
//==============================================================================

enum class Compilation { Debug, Release };
#ifdef NDEBUG
static constexpr Compilation compilation = Compilation::Release;
#else
static constexpr Compilation compilation = Compilation::Debug;
#endif

//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
