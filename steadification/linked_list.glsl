#ifndef TATOOINE_LINKED_LIST
#define TATOOINE_LINKED_LIST
//------------------------------------------------------------------------------
#include "node.glsl"
//------------------------------------------------------------------------------
layout(binding = 0, std430) buffer frontbind {
  node front_buffer[];
};
layout(binding = 1, std430) buffer backbind {
  node back_buffer[];
};
layout(binding = 2, std430) buffer sizebind {
  uint size_buffer[];
};
//------------------------------------------------------------------------------
uniform uvec2 resolution;

//==============================================================================
void insert(ivec2 texpos, vec2 v, float t, float t0, float curvature,
            uint render_index, uint layer) {
  const uint i = atomicCounterIncrement(ll_cnt);
  if (i < ll_size) {
    const uint len = imageAtomicAdd(ll_list_length_tex, texpos, 1);
    ll_nodes[i].next_index = imageAtomicExchange(ll_head_index_tex, texpos, i);
    ll_nodes[i].v          = v;
    ll_nodes[i].t          = t;
    ll_nodes[i].t0         = t0;
    ll_nodes[i].curvature  = curvature;
    ll_nodes[i].render_index = render_index;
    ll_nodes[i].layer        = layer;
  }
}
//------------------------------------------------------------------------------
node front(uint x, uint y) { return front_buffer[x + resolution.x * y]; }
node front(uvec2 xy) { return front(xy.x, xy.y); }
//------------------------------------------------------------------------------
node back(uint x, uint y) { return back_buffer[x + resolution.x * y]; }
node back(uvec2 xy) { return back(xy.x, xy.y); }
//------------------------------------------------------------------------------
uint size(uint x, uint y) { return size_buffer[x + resolution.x * y]; }
uint size(uvec2 xy) { return size(xy.x, xy.y); }
#endif
