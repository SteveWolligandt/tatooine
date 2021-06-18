#ifndef __YAVIN_TYPE_H__
#define __YAVIN_TYPE_H__

#include "glincludes.h"

//==============================================================================
namespace yavin {
//==============================================================================
enum Type {
  UBYTE          = GL_UNSIGNED_BYTE,
  BYTE           = GL_BYTE,
  USHORT         = GL_UNSIGNED_SHORT,
  SHORT          = GL_SHORT,
  UINT           = GL_UNSIGNED_INT,
  INT            = GL_INT,
  FLOAT          = GL_FLOAT,
  UBYTE332       = GL_UNSIGNED_BYTE_3_3_2,
  UBYTE233REV    = GL_UNSIGNED_BYTE_2_3_3_REV,
  USHORT565      = GL_UNSIGNED_SHORT_5_6_5,
  USHORT565REV   = GL_UNSIGNED_SHORT_5_6_5_REV,
  USHORT4444     = GL_UNSIGNED_SHORT_4_4_4_4,
  USHORT4444REV  = GL_UNSIGNED_SHORT_4_4_4_4_REV,
  USHORT5551     = GL_UNSIGNED_SHORT_5_5_5_1,
  USHORT1555REV  = GL_UNSIGNED_SHORT_1_5_5_5_REV,
  UINT8888       = GL_UNSIGNED_INT_8_8_8_8,
  UINT8888REV    = GL_UNSIGNED_INT_8_8_8_8_REV,
  UINT1010102    = GL_UNSIGNED_INT_10_10_10_2,
  UINT2101010REV = GL_UNSIGNED_INT_2_10_10_10_REV
};
//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
