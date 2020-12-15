#ifndef TATOOINE_FLOWEXPLORER_DIRECTORIES_H
#define TATOOINE_FLOWEXPLORER_DIRECTORIES_H
//==============================================================================
#include <filesystem>
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
auto fonts_directory() -> std::filesystem::path;
auto icons_directory() -> std::filesystem::path;
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
#endif
