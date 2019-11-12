#version 450

#include "streamsurface.glsl"
layout(location = 0) out vec4 vector;

void main() {
  if (!tau_in_range()) discard;
  vector = vec4(vf_frag, uv_frag.y, 1);
}
