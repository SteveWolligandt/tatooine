#include "yavin/atomiccounterbuffer.h"

#include <yavin/glincludes.h>

#include <cstring>
//==============================================================================
namespace yavin {
//==============================================================================
atomiccounterbuffer::atomiccounterbuffer(usage_t usage) : buffer(usage) {}
//------------------------------------------------------------------------------
atomiccounterbuffer::atomiccounterbuffer(size_t n, usage_t usage)
    : buffer(n, usage) {}
//------------------------------------------------------------------------------
atomiccounterbuffer::atomiccounterbuffer(size_t n, GLuint initial,
                                         usage_t usage)
    : buffer(n, initial, usage) {}
//------------------------------------------------------------------------------
atomiccounterbuffer::atomiccounterbuffer(const std::vector<GLuint>& data,
                                         usage_t                    usage)
    : buffer(data, usage) {}
//------------------------------------------------------------------------------
atomiccounterbuffer::atomiccounterbuffer(std::initializer_list<GLuint>&& data)
    : buffer(std::move(data), default_usage) {}
//------------------------------------------------------------------------------
void atomiccounterbuffer::set_all_to(GLuint val) {
  auto gpu_data = reinterpret_cast<unsigned char*>(
      gl::map_named_buffer(this->id(), GL_WRITE_ONLY));
  gl_error_check("glMapNamedBuffer");
  std::memset(gpu_data, val, sizeof(GLuint) * size());
  gl::unmap_named_buffer(this->id());
}
//------------------------------------------------------------------------------
void atomiccounterbuffer::bind(size_t i) const {
  gl::bind_buffer_base(GL_ATOMIC_COUNTER_BUFFER, i, this->id());
}
//------------------------------------------------------------------------------
void atomiccounterbuffer::unbind(size_t i) {
  gl::bind_buffer_base(GL_ATOMIC_COUNTER_BUFFER, i, 0);
}
//==============================================================================
}  // namespace yavin
//==============================================================================
