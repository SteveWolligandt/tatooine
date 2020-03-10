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
uniform float shininess;
uniform float ambient_factor;
uniform float diffuse_factor;
uniform float specular_factor;
uniform bool  animate;
uniform float time;
//------------------------------------------------------------------------------
const float min_alpha = 0.05;
//------------------------------------------------------------------------------
void main() {
  if (animate) {
    if (time > frag_parameterization) {
      out_color.a =
          clamp((1 - abs(time - frag_parameterization)), min_alpha, 1.0);
    } else {
      out_color.a = min_alpha;
    }
  } else {
    out_color.a = 1;
  }
  if (line_width - abs(frag_contour_parameterization) < contour_width) {
    out_color.rgb = vec3(0); return;
  } 
  vec3  V         = normalize(frag_position);
  vec3  L         = V;
  vec3  T         = normalize(frag_tangent);
  float LT        = dot(L, T);
  float VT        = dot(V, T);
  float diffuse   = clamp(sqrt(1 - LT * LT), 0, 1);
  float specular  = clamp(pow(
      LT * VT - sqrt(1 - LT * LT) * sqrt(1 - VT * VT), shininess), 0,1);
  out_color.rgb = vec3(0);
  out_color.rgb += ambient_factor * color;
  out_color.rgb += diffuse_factor * diffuse * color;
  out_color.rgb += specular_factor * specular * vec3(1);
}
