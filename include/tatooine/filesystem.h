#ifdef TATOOINE_STD_FILESYSTEM_AVAILABLE
#include <filesystem>
#else
#include <boost/filesystem.hpp>
#endif
//==============================================================================
namespace tatooine {
//==============================================================================
#ifdef TATOOINE_STD_FILESYSTEM_AVAILABLE
namespace filesystem = std::filesystem;
#else
namespace filesystem = boost::filesystem;
#endif
//==============================================================================
} // namespace tatooine
//==============================================================================
