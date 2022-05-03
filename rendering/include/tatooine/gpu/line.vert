#version 330 core
//------------------------------------------------------------------------------
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 tangent;
layout(location = 2) in float parameterization;
//------------------------------------------------------------------------------
out vec3  geom_position;
out vec3  geom_tangent;
out float geom_parameterization;
//------------------------------------------------------------------------------
void main() {
  geom_position         = position;
  geom_tangent          = tangent;
  geom_parameterization = parameterization;
}
