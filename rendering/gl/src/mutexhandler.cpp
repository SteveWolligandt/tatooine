#include <yavin/mutexhandler.h>

//==============================================================================
namespace yavin::detail {
//==============================================================================

std::mutex mutex::buffer;
std::mutex mutex::gl_call;

//==============================================================================
}  // namespace yavin::detail
//==============================================================================
