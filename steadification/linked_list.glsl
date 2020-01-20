#ifndef LINKED_LIST
#define LINKED_LIST
//------------------------------------------------------------------------------
struct node {
  vec2 pos;
  vec2 v;
  vec2 uv;
  uint next;
  float _padding;
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
    uint len = imageAtomicAdd(ll_head_index_tex, ivec2(gl_FragCoord.xy), 1);
    if (len > 2) { return; }
    uint next_index =
        imageAtomicExchange(ll_head_index_tex, ivec2(gl_FragCoord.xy), i);
    ll_nodes[i].pos  = pos_frag;
    ll_nodes[i].v    = v_frag;
    ll_nodes[i].uv   = uv_frag;
    ll_nodes[i].next = next_index;
  }
}
//------------------------------------------------------------------------------
#endif
