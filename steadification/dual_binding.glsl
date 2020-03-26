#ifndef TATOOINE_STEADIFICATION_DUAL_BINDING_GLSL
#define TATOOINE_STEADIFICATION_DUAL_BINDING_GLSL
//------------------------------------------------------------------------------
#include "node.glsl"
//------------------------------------------------------------------------------
layout(binding = 0, std430) buffer rast0_buffer {
  node rast0[];
};
layout(binding = 1, std430) buffer rast1_buffer {
  node rast1[];
};
//------------------------------------------------------------------------------
#endif
