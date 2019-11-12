#version 450

//------------------------------------------------------------------------------
#include "layouts.glsl"
#include "streamsurface.glsl"

//------------------------------------------------------------------------------
uniform uint size;

//------------------------------------------------------------------------------
void main() {
  if (tau_in_range()) {
    uint i = atomicCounterIncrement(cnt);
    if (i < size) {
      uint next_index
        = imageAtomicExchange(head_index_tex, ivec2(gl_FragCoord.xy), i);
      nodes[i].next = next_index;
      nodes[i].tau  = uv_frag.y;
      nodes[i].vf   = vf_frag;
    }
  }
}

