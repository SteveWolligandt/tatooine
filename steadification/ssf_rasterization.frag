#version 450
//==============================================================================
in vec2  v_frag;
in float t_frag;
in float t0_frag;
in float curvature_frag;
//==============================================================================
uniform uint render_index;
uniform uint layer;
uniform bool count;
uniform uint width;
//==============================================================================
layout(binding = 3, std430) buffer working_size_buffer {
  uint working_list_size[];
};
//==============================================================================
out vec4  v_t_t0_out;
out float curvature_out;
out uvec2 renderindex_layer_out;
//==============================================================================
void main() {
  v_t_t0_out            = vec4(v_frag, t_frag, t0_frag);
  curvature_out         = curvature_frag;
  renderindex_layer_out = uvec2(render_index, layer);
    const uint i = uint(floor(gl_FragCoord.x) + floor(gl_FragCoord.y) * width);
    atomicAdd(working_list_size[i], 1);
}
