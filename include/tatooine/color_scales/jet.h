#ifndef TATOOINE_COLOR_SCALES_JET_H
#define TATOOINE_COLOR_SCALES_JET_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>
#if TATOOINE_GL_AVAILABLE
#include <tatooine/gl/texture.h>
#endif

#include <memory>
//==============================================================================
namespace tatooine::color_scales {
//==============================================================================
template <floating_point Real>
struct jet  {
  using real_type  = Real;
  using this_type  = jet<Real>;
  using color_t = vec<Real, 3>;
  //==============================================================================
  static constexpr auto num_samples = std::size_t(7);
  //==============================================================================
  std::unique_ptr<color_t[]> m_samples;
  std::unique_ptr<real_type[]> m_parameterization;
  //==============================================================================
  jet()
      : m_samples{new color_t[]{color_t{0, 0, 0.5625}, color_t{0, 0, 1},
                             color_t{0, 1, 1}, color_t{0.5, 1, 0.5},
                             color_t{1, 1, 0}, color_t{1, 0, 0},
                             color_t{0.5, 0, 0}}},
        m_parameterization{new real_type[]{0, 0.111111, 0.36508, 0.492063,
                                        0.619047, 0.873016, 1}} {}
  //----------------------------------------------------------------------------
  auto sample(real_type t) const {
    if (t <= 0) {
      return m_samples[0];
    }
    if (t >= 1) {
      return color_t{m_samples[6]};
    }
    auto i = std::size_t{};
    for (; i < num_samples - 1; ++i) {
      if (m_parameterization[i] <= t && m_parameterization[i + 1] >= t) {
        t = (t - m_parameterization[i]) /
            (m_parameterization[i + 1] - m_parameterization[i]);
        break;
      }
    }
    return m_samples[i] * (1 - t) + m_samples[i + 1] * t;
  }
  auto operator()(real_type const t) const { return sample(t); }
  //----------------------------------------------------------------------------
#if TATOOINE_GL_AVAILABLE
  auto to_gpu_tex() {
    auto tex = gl::tex1rgb32f{m_samples.get(), num_samples};
    tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
    return tex;
  }
 //----------------------------------------------------------------------------
  auto to_gpu_tex2d(size_t const height = 2) {
    auto tex_data = std::vector<float>(num_samples * 3 * height);
    for (size_t i = 0; i < num_samples; ++i) {
      for (size_t j = 0; j < height; ++j) {
        tex_data[i * 3 + num_samples * j]     = m_samples[i](0);
        tex_data[i * 3 + 1 + num_samples * j] = m_samples[i](1);
        tex_data[i * 3 + 2 + num_samples * j] = m_samples[i](2);
      }
    }
    auto tex = gl::tex2rgb32f{tex_data, num_samples, height};
    tex.set_wrap_mode(gl::CLAMP_TO_EDGE);
    return tex;
  }
#endif
};
//==============================================================================
jet()->jet<double>;
//==============================================================================
}  // namespace tatooine::color_scales
//==============================================================================
#endif

