#include <tatooine/gl/glincludes.h>
#include <tatooine/gl/indexbuffer.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
indexbuffer::indexbuffer(buffer_usage usage) : buffer(usage) {}
//------------------------------------------------------------------------------
indexbuffer::indexbuffer(const indexbuffer& other) : buffer(other) {}
//------------------------------------------------------------------------------
indexbuffer::indexbuffer(indexbuffer&& other) : buffer(std::move(other)) {}
//------------------------------------------------------------------------------
indexbuffer& indexbuffer::operator=(const indexbuffer& other) {
  parent_type::operator=(other);
  return *this;
}
//------------------------------------------------------------------------------
indexbuffer& indexbuffer::operator=(indexbuffer&& other) {
  parent_type::operator=(std::move(other));
  return *this;
}
//------------------------------------------------------------------------------
indexbuffer::indexbuffer(GLsizei n, buffer_usage usage) : buffer(n, usage) {}
//------------------------------------------------------------------------------
indexbuffer::indexbuffer(GLsizei n, unsigned int initial, buffer_usage usage)
    : buffer(n, initial, usage) {}
//------------------------------------------------------------------------------
indexbuffer::indexbuffer(const std::vector<unsigned int>& data,
                         buffer_usage                     usage)
    : buffer(data, usage) {}
//------------------------------------------------------------------------------
indexbuffer::indexbuffer(std::initializer_list<unsigned int>&& list)
    : buffer(std::move(list), default_usage) {}
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
