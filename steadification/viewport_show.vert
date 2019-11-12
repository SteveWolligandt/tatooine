#version 450

uniform mat4 projection;

layout(location = 0) in vec2 vertex;

void main() {
  gl_Position = projection * vec4(vertex, 0, 1);
}

