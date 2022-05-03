/// This wrapper is needed in case that std::filesystem is not available.
/// This is the case for gcc 7.
#if TATOOINE_STD_FILESYSTEM_AVAILABLE
#include <filesystem>
#else
#include <boost/filesystem.hpp>
#endif
//==============================================================================
namespace tatooine {
//==============================================================================
#if TATOOINE_STD_FILESYSTEM_AVAILABLE
namespace filesystem = std::filesystem;
#else
namespace filesystem = boost::filesystem;
#endif
//==============================================================================
} // namespace tatooine
//==============================================================================
