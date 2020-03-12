#version 450
//==============================================================================
in vec2 pos_frag;
in vec2 v_frag;
in float tau_frag;
in float curvature_frag;
//==============================================================================
uniform uint render_index;
//==============================================================================
#include "linked_list.glsl"
//==============================================================================
void main() {
  const ivec2 texpos = ivec2(floor(gl_FragCoord.x), floor(gl_FragCoord.y));
  ll_push_back(texpos, pos_frag, v_frag, tau_frag, curvature_frag, render_index);
}
