#ifndef __YAVIN_INDEXBUFFER_H__
#define __YAVIN_INDEXBUFFER_H__

#include <initializer_list>
#include <iostream>
#include <vector>
#include "buffer.h"
#include "dllexport.h"
#include "errorcheck.h"

//==============================================================================
namespace yavin {
//==============================================================================

class indexbuffer : public buffer<GL_ELEMENT_ARRAY_BUFFER, unsigned int> {
 public:
  using parent_t = buffer<GL_ELEMENT_ARRAY_BUFFER, unsigned int>;
  using this_t   = indexbuffer;

  static const usage_t default_usage = STATIC_DRAW;

  DLL_API indexbuffer(usage_t usage = default_usage);
  DLL_API indexbuffer(const indexbuffer& other);
  DLL_API indexbuffer(indexbuffer&& other);
  DLL_API this_t& operator=(const this_t& other);
  DLL_API this_t& operator=(this_t&& other);

  DLL_API indexbuffer(size_t n, usage_t usage = default_usage);
  DLL_API indexbuffer(size_t n, unsigned int initial,
                      usage_t usage = default_usage);
  DLL_API indexbuffer(const std::vector<unsigned int>& data,
                      usage_t                          usage = default_usage);
  DLL_API indexbuffer(std::initializer_list<unsigned int>&& list);
};

//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
