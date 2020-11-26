#version 330 core
//------------------------------------------------------------------------------
layout(points) in;
layout(line_strip, max_vertices = 100) out;
//------------------------------------------------------------------------------
in mat4 model_matrix[1];
//------------------------------------------------------------------------------
uniform mat4  view_projection_matrix;
//------------------------------------------------------------------------------
void main() {
  const int num_vertices = 33;
  const float normalizer  = 1 / float(num_vertices - 1) * 2 * 3.14159;
  mat4 MVP = view_projection_matrix * model_matrix[0];
  vec4 center = MVP * vec4(0,0,0,1);

  // culling
  float       threshold    = 5;
  if (center.x < -threshold || center.x > threshold ||
      center.y < -threshold || center.y > threshold) {
    return;
  }
  for (int i = 0; i < num_vertices; ++i) {
    vec4 circle_pos =
        vec4(cos(float(i) * normalizer), sin(float(i) * normalizer), 0, 1);
    gl_Position = MVP * circle_pos;
    EmitVertex();
  }
  EndPrimitive();
}
