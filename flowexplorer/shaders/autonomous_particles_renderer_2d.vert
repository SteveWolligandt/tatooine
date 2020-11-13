#version 450
//==============================================================================
layout(location = 0) in vec2 mat_col0;
layout(location = 1) in vec2 mat_col1;
layout(location = 2) in vec2 mat_col2;
//==============================================================================
out mat4 model_matrix;
//==============================================================================
void main() {
  model_matrix = mat4(vec4(mat_col0, 0, 0),
                      vec4(mat_col1, 0, 0),
                      vec4(0),
                      vec4(mat_col2, 0, 1));
}
