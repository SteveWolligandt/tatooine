#version 450
//==============================================================================
in vec2 pos_frag;
in vec2 v_frag;
in vec2 uv_frag;
//==============================================================================
layout(location = 0) out vec2 v_out;
layout(location = 1) out vec2 uv_out;
//==============================================================================
void main() {
  //v_out  = v_frag;
  //uv_out = uv_frag;
  v_out  = vec2(0,0);
  uv_out = vec2(1,1);
}
