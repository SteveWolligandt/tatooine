#version 450

uniform sampler2D tex;
in vec3 pos_frag;
in vec2 uv_frag;

layout(location = 0) out vec4 frag_color;

void main() {
  frag_color = vec4(texture(tex, uv_frag).r);
  // frag_color = vec4(1,1,1,1);
}
