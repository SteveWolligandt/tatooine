#version 450

#include "layouts.glsl"
const uint END = 0xffffffff;
const float MAX_FLOAT = 3.402823466e+38;

layout(location = 0) out vec4 frag_color;

uniform float t0;
uniform float bw_tau;
uniform float fw_tau;

void main() {
  ivec2 pos  = ivec2(gl_FragCoord.xy);

  const vec3 head_vec_data = imageLoad(head_vectors_tex, pos).xyz;
  const vec2 head_vec      = head_vec_data.xy;
  const bool has_head_vec  = head_vec_data.z == 1;
  const uint head_index    = imageLoad(head_index_tex, pos).r;
  uint       running_index = head_index;
  //uint       front_index   = END;
  uint       len           = 0;
  float      min_tau       = MAX_FLOAT;

  while (running_index != END) {
    if (nodes[running_index].tau < min_tau) {
      //front_index = running_index;
      min_tau = nodes[running_index].tau;
    }
    running_index = nodes[running_index].next;
    ++len;
  }

  if (len == 0) {
    if (has_head_vec) {
      frag_color.xyz = vec3(0,0,1);
    } else {
      frag_color.xyz = vec3(0);
    }

  } else if (len == 1) {
    if (has_head_vec) {
      float cosalpha = dot(
           normalize(nodes[head_index].vf),
           normalize(head_vec));
      frag_color.xyz = vec3(0,0,1 - cosalpha*cosalpha);
    } else {
      frag_color.xyz = vec3(0.5);
      float tau = nodes[head_index].tau;
      float b = 0;
      if (tau < 0) { b = tau / bw_tau; }
      else { b = tau / fw_tau; }
      frag_color.xyz += vec3(b/2);
    }

  } else if (len == 2) {
    if (has_head_vec) {
      frag_color.xyz = vec3(1,0,1);
    } else {
      float cosalpha = dot(
           normalize(nodes[head_index].vf),
           normalize(nodes[nodes[head_index].next].vf));
      frag_color.xyz = vec3(1 - cosalpha*cosalpha);
    }
  } else {
    frag_color.xyz = vec3(1,0,0);
  }
}
