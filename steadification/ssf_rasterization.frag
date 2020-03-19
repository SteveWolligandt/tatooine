#version 450
//==============================================================================
in vec2 v_frag;
in float t_frag;
in float t0_frag;
in float curvature_frag;
//==============================================================================
uniform uint render_index;
uniform uint layer;
//==============================================================================
#include "linked_list.glsl"
//==============================================================================
void main() {
  const ivec2 texpos = ivec2(floor(gl_FragCoord.x), floor(gl_FragCoord.y));
  ll_push_back(texpos, v_frag, t_frag, t0_frag, curvature_frag, render_index,
               layer);
}
