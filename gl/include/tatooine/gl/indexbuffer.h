#ifndef TATOOINE_GL_INDEXBUFFER_H
#define TATOOINE_GL_INDEXBUFFER_H
//==============================================================================
#include <initializer_list>
#include <iostream>
#include <vector>
#include "buffer.h"
#include "dllexport.h"
#include "errorcheck.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class indexbuffer : public buffer<GL_ELEMENT_ARRAY_BUFFER, unsigned int> {
 public:
  using parent_type = buffer<GL_ELEMENT_ARRAY_BUFFER, unsigned int>;
  using this_type   = indexbuffer;

  static const auto default_usage = buffer_usage::STATIC_DRAW;

  DLL_API indexbuffer(buffer_usage usage = default_usage);
  DLL_API indexbuffer(const indexbuffer& other);
  DLL_API indexbuffer(indexbuffer&& other);
  DLL_API this_type& operator=(const this_type& other);
  DLL_API this_type& operator=(this_type&& other);

  DLL_API indexbuffer(GLsizei n, buffer_usage usage = default_usage);
  DLL_API indexbuffer(GLsizei n, unsigned int initial,
                      buffer_usage usage = default_usage);
  DLL_API indexbuffer(const std::vector<unsigned int>& data,
                      buffer_usage                          usage = default_usage);
  DLL_API indexbuffer(std::initializer_list<unsigned int>&& list);
};

//==============================================================================
}  // namespace tatooine::gl
//==============================================================================

#endif
