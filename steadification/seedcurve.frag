#version 450
//==============================================================================
layout(location = 0) out vec4 fragout;
uniform sampler2D color_scale;
uniform float min_t;
uniform float max_t;
uniform bool use_color_scale;
uniform vec4 color;
in float t0_frag;
//==============================================================================
void main() {
  if (use_color_scale) {
    fragout.rgb =
        texture(color_scale, 1 - vec2((t0_frag - min_t) / (max_t - min_t), 0.5))
            .rgb;
    fragout.a = 1;
  } else {
    fragout = color;
  }

}
