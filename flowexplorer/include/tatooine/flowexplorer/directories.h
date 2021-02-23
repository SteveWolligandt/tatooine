#ifndef TATOOINE_FLOWEXPLORER_DIRECTORIES_H
#define TATOOINE_FLOWEXPLORER_DIRECTORIES_H
//==============================================================================
#include <tatooine/filesystem.h>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
auto fonts_directory() -> filesystem::path;
auto icons_directory() -> filesystem::path;
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
