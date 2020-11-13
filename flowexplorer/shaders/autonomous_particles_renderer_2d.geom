#version 330 core
//------------------------------------------------------------------------------
layout(points) in;
layout(line_strip, max_vertices = 23) out;
//------------------------------------------------------------------------------
in mat4 model_matrix[1];
//------------------------------------------------------------------------------
uniform mat4  view_projection_matrix;
//------------------------------------------------------------------------------
void main() {
  const int num_vertices = 23;
  const float normalizer  = 1 / float(num_vertices - 1) * 2 * 3.14159;
  for (int i = 0; i < num_vertices; ++i) {
    vec4 circle_pos =
        vec4(cos(float(i) * normalizer), sin(float(i) * normalizer), 0, 1);
    gl_Position = view_projection_matrix * model_matrix[0] * circle_pos;
    EmitVertex();
  }
  EndPrimitive();
}
