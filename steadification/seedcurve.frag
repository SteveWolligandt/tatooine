#version 450
//==============================================================================
layout(location = 0) out vec3 fragout;
uniform sampler2D color_scale;
uniform float min_t;
uniform float max_t;
in float t_frag;
//==============================================================================
void main() {
  fragout =
      texture(color_scale, 1 - vec2((t_frag - min_t) / (max_t - min_t), 0.5))
          .rgb;
}
