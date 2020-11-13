#version 330 core
//------------------------------------------------------------------------------
layout(points) in;
layout(triangle_strip, max_vertices = 1000) out;
//------------------------------------------------------------------------------
in mat4 model_matrix[1];
//------------------------------------------------------------------------------
uniform mat4 view_projection_matrix;
//------------------------------------------------------------------------------
const float X = 0.525731112119133606;
const float Z = 0.850650808352039932;
void        main() {
  vec4 vs[] = vec4[](vec4(-X, 0, Z, 1), vec4(X, 0, Z, 1), vec4(-X, 0, -Z, 1),
                     vec4(X, 0, -Z, 1), vec4(0, Z, X, 1), vec4(0, Z, -X, 1),
                     vec4(0, -Z, X, 1), vec4(0, -Z, -X, 1), vec4(Z, X, 0, 1),
                     vec4(-Z, X, 0, 1), vec4(Z, -X, 0, 1), vec4(-Z, -X, 0, 1));

  mat4 MVP = view_projection_matrix * model_matrix[0];

  int indices[] =
      int[](0, 4, 1, 0, 9, 4, 9, 5, 4, 4, 5, 8, 4, 8, 1, 8, 10, 1, 8, 3, 10, 5,
            3, 8, 5, 2, 3, 2, 7, 3, 7, 10, 3, 7, 6, 10, 7, 11, 6, 11, 0, 6, 0,
            1, 6, 6, 1, 10, 9, 0, 11, 9, 11, 2, 9, 2, 5, 7, 2, 11);

  for (int i = 0; i < indices.length(); i += 3) {
    vec4 v01 = vs[indices[i]] + vs[indices[i + 1]];
    v01.w    = 1;
    vec4 v02 = vs[indices[i]] + vs[indices[i + 2]];
    v02.w    = 1;
    vec4 v12 = vs[indices[i + 1]] + vs[indices[i + 2]];
    v12.w    = 1;

    v01.xyz = normalize(v01.xyz / 2);
    v02.xyz = normalize(v02.xyz / 2);
    v12.xyz = normalize(v12.xyz / 2);

    gl_Position = MVP * vs[indices[i]];
    EmitVertex();
    gl_Position = MVP * v01;
    EmitVertex();
    gl_Position = MVP * v02;
    EmitVertex();
    gl_Position = MVP * v12;
    EmitVertex();
    gl_Position = MVP * vs[indices[i + 2]];
    EmitVertex();
    EndPrimitive();

    gl_Position = MVP * v01;
    EmitVertex();
    gl_Position = MVP * vs[indices[i + 1]];
    EmitVertex();
    gl_Position = MVP * v12;
    EmitVertex();
    EndPrimitive();
  }
}
