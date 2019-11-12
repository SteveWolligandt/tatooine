#version 450

uniform mat4 projection;

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec2 vf;

out vec2 pos_frag;
out vec2 uv_frag;
out vec2 vf_frag;

void main() {
  gl_Position = projection * vec4(pos, -uv.y, 1);
  pos_frag    = pos;
  uv_frag     = uv;
  vf_frag     = vf;
}

