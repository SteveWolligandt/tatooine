#version 330 core
//------------------------------------------------------------------------------
in vec3  frag_position;
in vec3  frag_tangent;
in float frag_parameterization;
in float frag_contour_parameterization;
//------------------------------------------------------------------------------
out vec4 out_color;
//------------------------------------------------------------------------------
uniform vec3  color;
uniform float line_width;
uniform float contour_width;
//------------------------------------------------------------------------------
void main() {
  if (line_width - abs(frag_contour_parameterization) < contour_width) {
    out_color = vec4(vec3(0),1); return;
  } 
  vec3  V         = normalize(frag_position);
  vec3  L         = V;
  vec3  T         = normalize(frag_tangent);
  float LT        = dot(L, T);
  float VT        = dot(V, T);
  float shininess = 10;
  float diffuse   = clamp(sqrt(1 - LT * LT), 0, 1);
  float specular  = clamp(pow(
      LT * VT - sqrt(1 - LT * LT) * sqrt(1 - VT * VT), shininess), 0,1);
  out_color.rgb = vec3(0);
  out_color.rgb += 0.5 * color;
  out_color.rgb += 0.5 * diffuse * color;
  out_color.rgb += specular * vec3(1);
  out_color.a = 1;
}
