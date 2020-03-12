#version 330 core
//------------------------------------------------------------------------------
in vec3  frag_position;
in vec3  frag_tangent;
in float frag_parameterization;
in float frag_contour_parameterization;
//------------------------------------------------------------------------------
out vec4 out_color;
//------------------------------------------------------------------------------
uniform vec3  line_color;
uniform vec3  contour_color;
uniform float line_width;
uniform float contour_width;
uniform float shininess;
uniform float ambient_factor;
uniform float diffuse_factor;
uniform float specular_factor;
uniform bool  animate;
uniform float fade_length;
uniform float time;
uniform float general_alpha;
uniform float animation_min_alpha;
//------------------------------------------------------------------------------
void main() {
  if (animate) {
    if (time > frag_parameterization) {
      out_color.a = clamp((1 - abs(time - frag_parameterization) / fade_length),
                          animation_min_alpha, 1.0);
    } else {
      out_color.a = animation_min_alpha;
    }
  } else {
    out_color.a = general_alpha;
  }
  if (line_width / 2 - abs(frag_contour_parameterization) < contour_width) {
    out_color.rgb = contour_color;
    return;
  }
  vec3  V         = normalize(frag_position);
  vec3  L         = V;
  vec3  T         = normalize(frag_tangent);
  float LT        = dot(L, T);
  float VT        = dot(V, T);
  float LN        = sqrt(1 - LT * LT);
  float VN        = sqrt(1 - VT * VT);
  float VR        = LT * VT - LN * VN;

  float diffuse   = VN;
  float specular  = pow(max(0,-VR), shininess);
  out_color.rgb = vec3(0);
  out_color.rgb += ambient_factor * line_color;
  out_color.rgb += diffuse_factor * diffuse * line_color;
  out_color.rgb += specular_factor * specular;
}
