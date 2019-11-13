uniform mat4 projection;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 vec;
layout(location = 2) in float tau;

out vec3 pos_frag;
out vec2 vec_frag;
out float tau_frag;

void main() {
  gl_Position = projection * vec4(pos, 1);
}
