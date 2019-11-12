#ifndef LAYOUTS
#define LAYOUTS

#include "node.glsl"

layout(binding = 0)          uniform atomic_uint cnt;

layout(binding = 0, r32ui)   uniform uimage2D head_index_tex;
layout(binding = 1, rgba32f) uniform image2D  head_vectors_tex;

layout(binding = 0, std430)          buffer   linked_list    { node nodes[]; };
layout(binding = 1, std430)          buffer   weights_buffer { float weights[]; };

#endif 
