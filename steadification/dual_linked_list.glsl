#ifndef LINKED_LIST
#define LINKED_LIST
//------------------------------------------------------------------------------
#include "end_index.glsl"
#include "node.glsl"
//------------------------------------------------------------------------------
uniform uint ll0_size;
uniform uint ll1_size;
layout(binding = 0) uniform atomic_uint ll0_cnt;
layout(binding = 0, r32ui) uniform uimage2D ll0_head_index_tex;
layout(binding = 1, r32ui) uniform uimage2D ll0_list_length_tex;
layout(binding = 0, std430) buffer ll0_data {
  node ll0_nodes[];
};

layout(binding = 1) uniform atomic_uint ll1_cnt;
layout(binding = 2, r32ui) uniform uimage2D ll1_head_index_tex;
layout(binding = 3, r32ui) uniform uimage2D ll1_list_length_tex;
layout(binding = 1, std430) buffer ll1_data {
  node ll1_nodes[];
};
//------------------------------------------------------------------------------
const ivec2 ll0_tex_resolution = imageSize(ll0_head_index_tex);
const ivec2 ll1_tex_resolution = imageSize(ll1_head_index_tex);
//------------------------------------------------------------------------------
void ll0_push_back(ivec2 texpos,  vec2 v, float t,float t0, float curvature,
                   uint render_index, uint layer) {
  const uint i = atomicCounterIncrement(ll0_cnt);
  if (i < ll0_size) {
    const uint len = imageAtomicAdd(ll0_list_length_tex, texpos, 1);
    // if (len > 2) { return; }
    ll0_nodes[i].next_index =
        imageAtomicExchange(ll0_head_index_tex, texpos, i);
    ll0_nodes[i].v            = v;
    ll0_nodes[i].t            = t;
    ll0_nodes[i].t0           = t0;
    ll0_nodes[i].curvature    = curvature;
    ll0_nodes[i].render_index = render_index;
    ll0_nodes[i].layer = layer;
  }
}
//------------------------------------------------------------------------------
void ll0_push_back(ivec2 texpos, node n) {
  const uint i = atomicCounterIncrement(ll0_cnt);
  if (i < ll0_size) {
    const uint len = imageAtomicAdd(ll0_list_length_tex, texpos, 1);
    // if (len > 2) { return; }
    ll0_nodes[i].next_index =
        imageAtomicExchange(ll0_head_index_tex, texpos, i);
    ll0_nodes[i].v            = n.v;
    ll0_nodes[i].t            = n.t;
    ll0_nodes[i].t0           = n.t0;
    ll0_nodes[i].curvature    = n.curvature;
    ll0_nodes[i].render_index = n.render_index;
    ll0_nodes[i].layer        = n.layer;
  }
}
//------------------------------------------------------------------------------
void ll1_push_back(ivec2 texpos, vec2 v, float t, float t0, float curvature,
                   uint render_index, uint layer) {
  const uint i = atomicCounterIncrement(ll1_cnt);
  if (i < ll1_size) {
    const uint len = imageAtomicAdd(ll1_list_length_tex, texpos, 1);
    ll1_nodes[i].next_index =
        imageAtomicExchange(ll1_head_index_tex, texpos, i);
    ll1_nodes[i].v            = v;
    ll1_nodes[i].t            = t;
    ll1_nodes[i].t0           = t0;
    ll1_nodes[i].curvature    = curvature;
    ll1_nodes[i].render_index = render_index;
    ll1_nodes[i].layer        = layer;
  }
}
//------------------------------------------------------------------------------
void ll1_push_back(ivec2 texpos, node n) {
  const uint i = atomicCounterIncrement(ll1_cnt);
  if (i < ll1_size) {
    const uint len = imageAtomicAdd(ll1_list_length_tex, texpos, 1);
    ll1_nodes[i].next_index =
        imageAtomicExchange(ll1_head_index_tex, texpos, i);
    ll1_nodes[i].v            = n.v;
    ll1_nodes[i].t            = n.t;
    ll1_nodes[i].t0           = n.t0;
    ll1_nodes[i].curvature    = n.curvature;
    ll1_nodes[i].render_index = n.render_index;
    ll1_nodes[i].layer        = n.layer;
  }
}
//------------------------------------------------------------------------------
node ll0_node_at(uint i) {
  return ll0_nodes[i];
}
//------------------------------------------------------------------------------
node ll1_node_at(uint i) {
  return ll1_nodes[i];
}
//------------------------------------------------------------------------------
uint ll0_head_index(ivec2 texpos) {
  return imageLoad(ll0_head_index_tex, texpos).r;
}
//------------------------------------------------------------------------------
uint ll1_head_index(ivec2 texpos) {
  return imageLoad(ll1_head_index_tex, texpos).r;
}
//------------------------------------------------------------------------------
node ll0_head_node(ivec2 texpos) {
  return ll0_node_at(ll0_head_index(texpos));
}
//------------------------------------------------------------------------------
node ll1_head_node(ivec2 texpos) {
  return ll1_node_at(ll1_head_index(texpos));
}
//------------------------------------------------------------------------------
uint ll0_size_at(ivec2 texpos) {
  return imageLoad(ll0_list_length_tex, texpos).r;
}
//------------------------------------------------------------------------------
uint ll0_size_at(ivec2 texpos, float min_btau, float max_ftau) {
  uint size = 0;
  const uint head_index = ll0_head_index(texpos);
  uint running_index = head_index;
  while (running_index != end_index) {
    node n = ll0_node_at(running_index);
    float tau = n.t - n.t0;
    if (min_btau <= tau && tau <= max_ftau) { ++size; }
    running_index = n.next_index;
  }
  return size;
}
//------------------------------------------------------------------------------
uint ll1_size_at(ivec2 texpos) {
  return imageLoad(ll1_list_length_tex, texpos).r;
}
//------------------------------------------------------------------------------
uint ll1_size_at(ivec2 texpos, float min_btau, float max_ftau) {
  uint size = 0;
  const uint head_index = ll1_head_index(texpos);
  uint running_index = head_index;
  while (running_index != end_index) {
    node n = ll1_node_at(running_index);
    float tau = n.t - n.t0;
    if (min_btau <= tau && tau <= max_ftau) { ++size; }
    running_index = n.next_index;
  }
  return size;
}
//------------------------------------------------------------------------------
node ll0_max_tau_node(ivec2 texpos) {
  const uint hi      = ll0_head_index(texpos);
  uint       ri      = hi;
  uint       mi      = hi;
  float      max_tau = -1e10;
  while (ri != end_index) {
    node n = ll0_node_at(ri);
    float tau    = n.t - n.t0;
    if (max_tau < tau) {
      max_tau = tau;
      mi      = ri;
    }
    ri = ll0_node_at(ri).next_index;
  }
  return ll0_node_at(mi);
}
//------------------------------------------------------------------------------
node ll1_max_tau_node(ivec2 texpos) {
  const uint hi      = ll1_head_index(texpos);
  uint       ri      = hi;
  uint       mi      = hi;
  float      max_tau = -1e10;
  while (ri != end_index) {
    node  n   = ll1_node_at(ri);
    float tau = n.t - n.t0;
    if (max_tau < tau) {
      max_tau = tau;
      mi      = ri;
    }
    ri = ll1_node_at(ri).next_index;
  }
  return ll1_node_at(mi);
}
//------------------------------------------------------------------------------
node ll0_min_tau_node(ivec2 texpos) {
  const uint hi      = ll0_head_index(texpos);
  uint       ri      = hi;
  uint       mi      = hi;
  float      min_tau = 1e10;
  while (ri != end_index) {
    node n = ll0_node_at(ri);
    float tau = n.t - n.t0;
    if (min_tau > tau) {
      min_tau = tau;
      mi      = ri;
    }
    ri = ll0_node_at(ri).next_index;
  }
  return ll0_node_at(mi);
}
//------------------------------------------------------------------------------
node ll1_min_tau_node(ivec2 texpos) {
  const uint hi      = ll1_head_index(texpos);
  uint       ri      = hi;
  uint       mi      = hi;
  float      min_tau = 1e10;
  while (ri != end_index) {
    node  n   = ll1_node_at(ri);
    float tau = n.t - n.t0;
    if (min_tau > tau) {
      min_tau = tau;
      mi      = ri;
    }
    ri = ll1_node_at(ri).next_index;
  }
  return ll1_node_at(mi);
}
#endif
