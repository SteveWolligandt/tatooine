#include <yavin/indexbuffer.h>

#include <yavin/glincludes.h>

//==============================================================================
namespace yavin {
//==============================================================================

indexbuffer::indexbuffer(usage_t usage) : buffer(usage) {}

//------------------------------------------------------------------------------

indexbuffer::indexbuffer(const indexbuffer& other) : buffer(other) {}

//------------------------------------------------------------------------------

indexbuffer::indexbuffer(indexbuffer&& other) : buffer(std::move(other)) {}

//------------------------------------------------------------------------------

indexbuffer& indexbuffer::operator=(const indexbuffer& other) {
  parent_t::operator=(other);
  return *this;
}

//------------------------------------------------------------------------------

indexbuffer& indexbuffer::operator=(indexbuffer&& other) {
  parent_t::operator=(std::move(other));
  return *this;
}

//------------------------------------------------------------------------------

indexbuffer::indexbuffer(size_t n, usage_t usage) : buffer(n, usage) {}

//------------------------------------------------------------------------------

indexbuffer::indexbuffer(size_t n, unsigned int initial, usage_t usage)
    : buffer(n, initial, usage) {}

//------------------------------------------------------------------------------

indexbuffer::indexbuffer(const std::vector<unsigned int>& data, usage_t usage)
    : buffer(data, usage) {}

//------------------------------------------------------------------------------

indexbuffer::indexbuffer(std::initializer_list<unsigned int>&& list)
    : buffer(std::move(list), default_usage) {}

//==============================================================================
}  // namespace yavin
//==============================================================================
