#version 450
//==============================================================================
in vec2 pos_frag;
in vec2 v_frag;
in vec2 uv_frag;
//==============================================================================
layout(location = 0) out vec2 pos_out;
layout(location = 1) out vec2 v_out;
layout(location = 2) out vec2 uv_out;

layout(binding = 0, r8) uniform image2D coverage_tex;
//==============================================================================
void main() {
  pos_out   = pos_frag;
  v_out     = v_frag;
  uv_out    = uv_frag;
  imageStore(coverage_tex, ivec2(floor(gl_FragCoord.x), floor(gl_FragCoord.y)),
             uvec4(1, 0, 0, 0));
}
