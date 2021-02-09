#version 330 core
//==============================================================================
uniform mat4 projection;
uniform mat4 modelview;
//==============================================================================
layout(location = 0) in vec3 pos;
//==============================================================================
void main() {
  gl_Position = projection * modelview * vec4(pos, 1);
  float threshold = 20;
  if (gl_Position.x < -threshold) {
    gl_Position.x = -1;
  }
  if (gl_Position.x > threshold) {
    gl_Position.x = 1;
  }
  if (gl_Position.y < -threshold) {
    gl_Position.y = -1;
  }
  if (gl_Position.y > threshold) {
    gl_Position.y = 1;
  }
}
