#version 450
//==============================================================================
layout(location = 0) in vec3 mat_col0;
layout(location = 1) in vec3 mat_col1;
layout(location = 2) in vec3 mat_col2;
layout(location = 3) in vec3 mat_col3;
//==============================================================================
out mat4 model_matrix;
//==============================================================================
void main() {
  model_matrix = mat4(vec4(mat_col0, 0),
                      vec4(mat_col1, 0),
                      vec4(mat_col2, 0),
                      vec4(mat_col3, 1));
}
