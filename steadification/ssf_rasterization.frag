#version 450
//==============================================================================
in vec2 v_frag;
in float t_frag;
in float t0_frag;
in float curvature_frag;
//==============================================================================
uniform uint render_index;
uniform uint layer;
uniform bool count;
//==============================================================================
layout(binding = 0, r32ui) uniform uimage2D list_size;
//==============================================================================
out vec2 v_out;
out vec2 t_t0_out;
out float curvature_out;
out uvec2 renderindex_layer_out;
//==============================================================================
void main() {
  v_out                 = v_frag;
  t_t0_out              = vec2(t_frag, t0_frag);
  curvature_out         = curvature_frag;
  renderindex_layer_out = uvec2(render_index, layer);
  if (count) {
    imageAtomicAdd(list_size, ivec2(floor(gl_FragCoord.xy)), 1);
  }
}
