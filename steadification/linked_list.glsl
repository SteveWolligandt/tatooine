#ifndef LINKED_LIST
#define LINKED_LIST
//------------------------------------------------------------------------------
#include "node.glsl"
#include "end_index.glsl"
//------------------------------------------------------------------------------
uniform uint ll_size;
layout(binding = 0) uniform atomic_uint ll_cnt;
layout(binding = 0, r32ui) uniform uimage2D ll_head_index_tex;
layout(binding = 1, r32ui) uniform uimage2D ll_list_length_tex;
layout(binding = 0, std430) buffer ll_data {
  node ll_nodes[];
};
//------------------------------------------------------------------------------
const ivec2 ll_tex_resolution = imageSize(ll_head_index_tex);
//------------------------------------------------------------------------------
void ll_push_back(ivec2 texpos, vec2 v, float t, float t0, float curvature,
                  uint render_index, uint layer) {
  const uint i = atomicCounterIncrement(ll_cnt);
  if (i < ll_size) {
    const uint len = imageAtomicAdd(ll_list_length_tex, texpos, 1);
    ll_nodes[i].next_index = imageAtomicExchange(ll_head_index_tex, texpos, i);
    ll_nodes[i].v          = v;
    ll_nodes[i].t            = t;
    ll_nodes[i].t0           = t0;
    ll_nodes[i].curvature  = curvature;
    ll_nodes[i].render_index = render_index;
    ll_nodes[i].layer        = layer;
  }
}
//------------------------------------------------------------------------------
uint ll_head_index(ivec2 texpos) {
  return imageLoad(ll_head_index_tex, texpos).r;
}
//------------------------------------------------------------------------------
node ll_node_at(uint i) {
  return ll_nodes[i];
}
//------------------------------------------------------------------------------
uint ll_size_at(ivec2 texpos) {
  return imageLoad(ll_list_length_tex, texpos).r;
}
//------------------------------------------------------------------------------
node ll_head_node(ivec2 texpos) {
  return ll_node_at(ll_head_index(texpos));
}
//------------------------------------------------------------------------------
node ll_min_tau_node(ivec2 texpos) {
  const uint hi        = ll_head_index(texpos);
  uint       ri        = hi;
  uint       mi        = hi;
  float      min_tau   = 1e10;
  uint       min_render_index = uint(0) -1;
  while (ri != end_index) {
    node n = ll_node_at(ri);
    if (min_render_index > n.render_index) {
      min_render_index = n.render_index;
      min_tau          = 1e10;
    }
    if (min_tau > n.t - n.t0) {
      min_tau = n.t - n.t0;
      mi      = ri;
    }
    ri = n.next_index;
  }
  return ll_node_at(mi);
}
//------------------------------------------------------------------------------
node ll_max_tau_node(ivec2 texpos) {
  const uint hi        = ll_head_index(texpos);
  uint       ri        = hi;
  uint       mi        = hi;
  float      max_tau   = -1e10;
  uint       min_render_index = 100000;
  while (ri != end_index) {
    node n = ll_node_at(ri);
    if (min_render_index > n.render_index) {
      min_render_index = n.render_index;
      max_tau          = -1e10;
    }
    if (max_tau < abs(n.t - n.t0)) {
      max_tau = abs(n.t - n.t0);
      mi      = ri;
    }
    ri = n.next_index;
  }
  return ll_node_at(mi);
}
#endif
