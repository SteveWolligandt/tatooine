#ifndef __YAVIN_MOVABLE_H__
#define __YAVIN_MOVABLE_H__

#include "dllexport.h"
#include "transform.h"

//==============================================================================
namespace yavin {
//==============================================================================

class movable {
 public:
  DLL_API movable()          = default;
  DLL_API virtual ~movable() = default;
  auto&       transform() { return m_transform; }
  const auto& transform() const { return m_transform; }

 protected:
  yavin::transform m_transform;
};

//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
