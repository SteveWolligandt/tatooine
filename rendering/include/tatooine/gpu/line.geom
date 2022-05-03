#version 330 core
//------------------------------------------------------------------------------
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;
//------------------------------------------------------------------------------
in vec3  geom_position[2];
in vec3  geom_tangent[2];
in float geom_parameterization[2];
//------------------------------------------------------------------------------
out vec3  frag_position;
out vec3  frag_tangent;
out float frag_parameterization;
out float frag_contour_parameterization;
//------------------------------------------------------------------------------
uniform mat4  projection_matrix;
uniform mat4  modelview_matrix;
uniform float line_width;
//------------------------------------------------------------------------------
void main() {
  vec3  p0mv     = (modelview_matrix * vec4(geom_position[0], 1)).xyz;
  vec3  p1mv     = (modelview_matrix * vec4(geom_position[1], 1)).xyz;
  vec3  tan0mv   = normalize((modelview_matrix * vec4(geom_tangent[0], 0)).xyz);
  vec3  tan1mv   = normalize((modelview_matrix * vec4(geom_tangent[1], 0)).xyz);
  vec3  p0par    = dot(-p0mv, tan0mv) * tan0mv + p0mv;
  vec3  p1par    = dot(-p1mv, tan1mv) * tan1mv + p1mv;
  vec3  y0       = normalize(cross(p0par, tan0mv));
  vec3  y1       = normalize(cross(p1par, tan1mv));
  float p0leninv = 1 / length(p0par);
  float p1leninv = 1 / length(p1par);
  vec3  x0       = normalize(cross(tan0mv, y0));
  vec3  x1       = normalize(cross(tan1mv, y1));
  float f0       = sqrt(1 - (line_width * p0leninv));
  float f1       = sqrt(1 - (line_width * p1leninv));
  vec3  e0p      = line_width * (line_width * p0leninv * x0 + f0 * y0);
  vec3  e0m      = line_width * (line_width * p0leninv * x0 - f0 * y0);
  vec3  e1p      = line_width * (line_width * p1leninv * x1 + f1 * y1);
  vec3  e1m      = line_width * (line_width * p1leninv * x1 - f1 * y1);
   //vec3  e0p     = line_width * y0;
   //vec3  e0m     = -line_width * y0;
   //vec3  e1p     = line_width * y1;
   //vec3  e1m     = -line_width * y1;

  frag_position                 = p0mv;
  frag_tangent                  = tan0mv;
  frag_parameterization         = geom_parameterization[0];
  frag_contour_parameterization = -line_width / 2;
  gl_Position                   = projection_matrix * vec4(p0mv + e0p, 1);
  EmitVertex();

  frag_position                 = p1mv;
  frag_tangent                  = tan1mv;
  frag_parameterization         = geom_parameterization[1];
  frag_contour_parameterization = -line_width / 2;
  gl_Position                   = projection_matrix * vec4(p1mv + e1p, 1);
  EmitVertex();

  frag_position                 = p0mv;
  frag_tangent                  = tan0mv;
  frag_parameterization         = geom_parameterization[0];
  frag_contour_parameterization = line_width / 2;
  gl_Position                   = projection_matrix * vec4(p0mv + e0m, 1);
  EmitVertex();

  frag_position                 = p1mv;
  frag_tangent                  = tan1mv;
  frag_parameterization         = geom_parameterization[1];
  frag_contour_parameterization = line_width / 2;
  gl_Position                   = projection_matrix * vec4(p1mv + e1m, 1);
  EmitVertex();

  EndPrimitive();
}
