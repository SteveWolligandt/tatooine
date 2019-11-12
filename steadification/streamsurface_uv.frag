#version 450

in vec2 uv_frag;
layout(location = 0) out vec2 frag_color;

void main() {
  frag_color = uv_frag;
}
