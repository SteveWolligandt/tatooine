#version 450
//==============================================================================
uniform mat4 projection;
//==============================================================================
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 v;
layout(location = 2) in float tau;
layout(location = 3) in float curvature;
//==============================================================================
out vec2  pos_frag;
out vec2  v_frag;
out float tau_frag;
out float curvature_frag;
//==============================================================================
void main() {
  gl_Position    = projection * vec4(pos, 0, 1);
  pos_frag       = pos;
  v_frag         = v;
  tau_frag       = tau;
  curvature_frag = curvature;
}
