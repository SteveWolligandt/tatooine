#version 330 core
//==============================================================================
uniform mat4 projection;
uniform mat4 modelview;
//==============================================================================
layout(location = 0) in vec3 pos;
//==============================================================================
out vec3 nor_frag;
//==============================================================================
void main() {
  mat4 MVP = projection * modelview;
  gl_Position = MVP * vec4(pos, 1);
  nor_frag = normalize((modelview * vec4(pos, 0)).xyz);
}
