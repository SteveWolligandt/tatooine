#version 430 core
//------------------------------------------------------------------------------
layout(location = 0) in vec2 vert_position;
layout(location = 1) in vec2 vert_uv;
//------------------------------------------------------------------------------
uniform mat4  projection_matrix;
uniform mat4  modelview_matrix;
//------------------------------------------------------------------------------
out vec2 frag_uv;
//------------------------------------------------------------------------------
void main() {
  gl_Position = projection_matrix * modelview_matrix * vec4(vert_position, 0, 1);
  frag_uv = vert_uv;
}
