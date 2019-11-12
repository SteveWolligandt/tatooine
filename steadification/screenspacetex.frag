#version 450

uniform sampler2D tex;
uniform uvec2 resolution;

layout(location = 0) out vec4 frag_color;

void main() {
  vec2 normalized_uv = vec2(gl_FragCoord.x / resolution.x,
                            gl_FragCoord.y / resolution.y);
  frag_color = texture(tex, normalized_uv);
  // frag_color = vec4(normalized_uv, 0,1);
}
