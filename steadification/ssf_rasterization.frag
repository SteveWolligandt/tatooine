#version 450
//==============================================================================
in vec2 pos_frag;
in vec2 v_frag;
in vec2 uv_frag;
in float curvature_frag;
//==============================================================================
#include "linked_list.glsl"
//==============================================================================
void main() {
  const ivec2 texpos = ivec2(floor(gl_FragCoord.x), floor(gl_FragCoord.y));
  ll_push_back(texpos, pos_frag, v_frag, uv_frag, curvature_frag);
}
