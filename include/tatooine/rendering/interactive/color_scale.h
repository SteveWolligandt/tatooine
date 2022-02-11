#ifndef TATOOINE_RENDERING_INTERACTIVE_COLOR_SCALE_H
#define TATOOINE_RENDERING_INTERACTIVE_COLOR_SCALE_H
//==============================================================================
#include <tatooine/color_scales/GYPi.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/gl/indexeddata.h>
#include <tatooine/gl/texture.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine::rendering::interactive {
//==============================================================================
struct color_scale {
  gl::tex1rgba32f tex;
  gl::tex2rgba32f tex_2d;

  template <typename ColorScale>
  color_scale(ColorScale&& c) {
    tex    = c.to_gpu_tex();
    tex_2d = to_2d(tex, 4);
  }
  static auto GYPi() -> auto& {
    static auto gypi = color_scale{tatooine::color_scales::GYPi<GLfloat>{}};
    return gypi;
  }
  static auto viridis() -> auto& {
    static auto viridis = color_scale{tatooine::color_scales::viridis<GLfloat>{}};
    return viridis;
  }
};
//==============================================================================
}  // namespace tatooine::rendering::interactive
//==============================================================================
#endif
