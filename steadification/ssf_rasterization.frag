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
layout(binding = 0, std430) buffer list_size_buffer {
  uint list_size[];
};
//==============================================================================
out vec4 v_t_t0_out;
out float curvature_out;
out uint width;
out uvec2 renderindex_layer_out;
//==============================================================================
void main() {
  v_t_t0_out            = vec4(v_frag, t_frag, t0_frag);
  curvature_out         = curvature_frag;
  renderindex_layer_out = uvec2(render_index, layer);
  if (count) {
    const uint i = uint(gl_FragCoord.x) + width * uint(gl_FragCoord.y);
    atomicAdd(list_size[i], 1);
  }
}
