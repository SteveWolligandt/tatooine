#ifndef LINKED_LIST
#define LINKED_LIST
//------------------------------------------------------------------------------
const uint end_index = 0xffffffff;
//------------------------------------------------------------------------------
struct node {
  vec2 pos;
  vec2 v;
  vec2 uv;
  uint next;
};                          
//------------------------------------------------------------------------------
uniform                             uint        ll_size;
layout(binding = 0)         uniform atomic_uint ll_cnt;
layout(binding = 0, r32ui)  uniform uimage2D    ll_head_index_tex;
layout(binding = 1, r32ui)  uniform uimage2D    ll_list_length_tex;
layout(binding = 0, std430)         buffer      ll_data{ node ll_nodes[]; };
//------------------------------------------------------------------------------
const ivec2 ll_tex_resolution = imageSize(ll_head_index_tex);
//------------------------------------------------------------------------------
void ll_push_back(ivec2 texpos, vec2 pos, vec2 v, vec2 uv) {
  const uint i = atomicCounterIncrement(ll_cnt);
  if (i < ll_size) {
    const uint len = imageAtomicAdd(ll_list_length_tex, texpos, 1);
    //if (len > 2) { return; }
    ll_nodes[i].next = imageAtomicExchange(ll_head_index_tex, texpos, i);
    ll_nodes[i].pos  = pos;
    ll_nodes[i].v    = v;
    ll_nodes[i].uv   = uv;
  }
}
//------------------------------------------------------------------------------
uint ll_head_index(ivec2 texpos) {
  return imageLoad(ll_head_index_tex, texpos).r;
}
//------------------------------------------------------------------------------
node ll_node_at(uint i) { return ll_nodes[i]; }
//------------------------------------------------------------------------------
uint ll_size_at(ivec2 texpos) {
  return imageLoad(ll_list_length_tex, texpos).r;
}
//------------------------------------------------------------------------------
node ll_head_node(ivec2 texpos) { return ll_node_at(ll_head_index(texpos)); }
//------------------------------------------------------------------------------
node ll_max_tau_node(ivec2 texpos) {
  const uint hi = ll_head_index(texpos);
  uint  ri      = hi;
  uint  mi      = hi;
  float max_tau = -1e10;
  while (ri != end_index) {
    if (max_tau < ll_node_at(ri).uv.y) {
      max_tau = ll_node_at(ri).uv.y;
      mi      = ri;
    }
    ri = ll_node_at(ri).next;
  }
  return ll_node_at(mi);
}
//------------------------------------------------------------------------------
node ll_min_tau_node(ivec2 texpos) {
  const uint hi = ll_head_index(texpos);
  uint  ri      = hi;
  uint  mi      = hi;
  float min_tau = 1e10;
  while (ri != end_index) {
    if (min_tau > ll_node_at(ri).uv.y) {
      min_tau = ll_node_at(ri).uv.y;
      mi      = ri;
    }
    ri = ll_node_at(ri).next;
  }
  return ll_node_at(mi);
}
#endif
