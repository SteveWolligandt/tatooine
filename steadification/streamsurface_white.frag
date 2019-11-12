#version 450

layout(location = 0) out vec4 frag_color;

in vec2 pos_frag;
in vec2 uv_frag;
in vec2 v_frag;

void main() {
  frag_color = vec4(1,1,1,1);
}
