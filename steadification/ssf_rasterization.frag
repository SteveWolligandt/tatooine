#version 450
//==============================================================================
in vec2 pos_frag;
in vec2 v_frag;
in vec2 uv_frag;
//==============================================================================
#include "linked_list.glsl"
//==============================================================================
layout(location = 0) out uint coverage_out;
//==============================================================================
void main() {
  const ivec2 texpos = ivec2(floor(gl_FragCoord.x), floor(gl_FragCoord.y));
  ll_push_back(texpos, pos_frag, v_frag, uv_frag);
  coverage_out = 255;
}
