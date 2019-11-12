#ifndef STREAMSURFACE
#define STREAMSURFACE

in vec2 pos_frag;
in vec2 uv_frag;
in vec2 vf_frag;

uniform vec2 backward_tau_range;
uniform vec2 forward_tau_range;
uniform vec2 u_range;

//------------------------------------------------------------------------------
vec2 taus() {
  float nu = (uv_frag.x - u_range.x) / (u_range.y - u_range.x);

  return vec2(mix(backward_tau_range.x, backward_tau_range.y, nu),
              mix(forward_tau_range.x, forward_tau_range.y , nu));
}

//------------------------------------------------------------------------------
bool tau_in_range() {
  vec2 forback_tau = taus();
  return forback_tau.x <= uv_frag.y && uv_frag.y <= forback_tau.y;
}

#endif
