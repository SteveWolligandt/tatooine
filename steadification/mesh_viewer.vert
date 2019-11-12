#version 450

uniform mat4 modelview;
uniform mat4 projection;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

out vec3 pos_frag;
out vec2 uv_frag;

void main() {
  gl_Position = projection * vec4(pos, 1);
  pos_frag = pos;
  uv_frag = uv;
}

