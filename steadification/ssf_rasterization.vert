#version 450
//==============================================================================
uniform mat4 projection;
uniform mat4 modelview;
//==============================================================================
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 v;
layout(location = 2) in vec2 uv;
//==============================================================================
out vec2 pos_frag;
out vec2 v_frag;
out vec2 uv_frag;
//==============================================================================
void main() {
  gl_Position = projection * vec4(pos, 0, 1);
  pos_frag    = pos;
  v_frag      = v;
  uv_frag     = uv;
}
