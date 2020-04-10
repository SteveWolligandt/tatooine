#version 450
//==============================================================================
layout(location = 0) out vec4 fragout;
uniform sampler2D color_scale;
uniform float min_t;
uniform float max_t;
uniform bool use_color_scale;
uniform vec3 color;
in float t_frag;
//==============================================================================
void main() {
  if (use_color_scale) {
    fragout.rgb =
        texture(color_scale, 1 - vec2((t_frag - min_t) / (max_t - min_t), 0.5))
            .rgb;
  } else {
    fragout.rgb = color;
  }
  fragout.a = 1;

}
