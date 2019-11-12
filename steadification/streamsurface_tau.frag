#version 450

#include "streamsurface.glsl"

uniform vec2 tau_range;
uniform sampler2D color_scale;

layout(location = 0) out vec4 frag_color;

void main() {
  if (!tau_in_range()) discard;

  float ntau = (uv_frag.y - tau_range.x) / (tau_range.y - tau_range.x);
  frag_color.xyz = texture(color_scale, vec2(ntau, 0.5)).rgb;
  frag_color.w = 1;
}
