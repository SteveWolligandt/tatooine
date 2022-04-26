#ifndef TATOOINE_RENDERING_INTERACTIVE_COLOR_SCALE_H
#define TATOOINE_RENDERING_INTERACTIVE_COLOR_SCALE_H
//==============================================================================
#include <tatooine/color_scales/BrBG.h>
#include <tatooine/color_scales/BuRD.h>
#include <tatooine/color_scales/GBBr.h>
#include <tatooine/color_scales/GYPi.h>
#include <tatooine/color_scales/GnRP.h>
#include <tatooine/color_scales/GnYIRd.h>
#include <tatooine/color_scales/OrPu.h>
#include <tatooine/color_scales/PRGn.h>
#include <tatooine/color_scales/PiYG.h>
#include <tatooine/color_scales/PuOr.h>
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
    auto gpu_data = std::vector<GLfloat>(4 * c.num_samples());
    for (std::size_t i = 0; i < c.num_samples(); ++i) {
      gpu_data[i * 4]     = c.data()[i](0);
      gpu_data[i * 4 + 1] = c.data()[i](1);
      gpu_data[i * 4 + 2] = c.data()[i](2);
      gpu_data[i * 4 + 3] = 1;
    }
    tex = gl::tex1rgba32f{gpu_data.data(), c.num_samples()};
    tex.set_wrap_mode(gl::wrap_mode::clamp_to_edge);
    tex_2d = to_2d(tex, 4);
  }
  static auto GYPi() -> auto& {
    static auto gypi = color_scale{tatooine::color_scales::GYPi<GLfloat>{}};
    return gypi;
  }
  static auto PiYG() -> auto& {
    static auto s = color_scale{tatooine::color_scales::PiYG<GLfloat>{}};
    return s;
  }
  static auto BrBG() -> auto& {
    static auto s = color_scale{tatooine::color_scales::BrBG<GLfloat>{}};
    return s;
  }
  static auto BuRD() -> auto& {
    static auto s = color_scale{tatooine::color_scales::BuRD<GLfloat>{}};
    return s;
  }
  static auto GBBr() -> auto& {
    static auto s = color_scale{tatooine::color_scales::GBBr<GLfloat>{}};
    return s;
  }
  static auto GnRP() -> auto& {
    static auto s = color_scale{tatooine::color_scales::GnRP<GLfloat>{}};
    return s;
  }
  static auto GnYIRd() -> auto& {
    static auto s = color_scale{tatooine::color_scales::GnYIRd<GLfloat>{}};
    return s;
  }
  static auto OrPu() -> auto& {
    static auto s = color_scale{tatooine::color_scales::OrPu<GLfloat>{}};
    return s;
  }
  static auto PRGn() -> auto& {
    static auto s = color_scale{tatooine::color_scales::PRGn<GLfloat>{}};
    return s;
  }
  static auto PuOr() -> auto& {
    static auto s = color_scale{tatooine::color_scales::PuOr<GLfloat>{}};
    return s;
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
