#version 450
//==============================================================================
uniform mat4 projection;
//==============================================================================
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 v;
layout(location = 2) in float t;
layout(location = 3) in float t0;
layout(location = 4) in float curvature;
//==============================================================================
out vec2  v_frag;
out float t_frag;
out float t0_frag;
out float curvature_frag;
//==============================================================================
void main() {
  gl_Position    = projection * vec4(pos, t - t0, 1);
  v_frag         = v;
  t_frag         = t;
  t0_frag        = t0;
  curvature_frag = curvature;
}
