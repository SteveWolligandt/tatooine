#ifndef TATOOINE_COLOR_SCALES_COOL_TO_WARM_H
#define TATOOINE_COLOR_SCALES_COOL_TO_WARM_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>

#include <memory>
//==============================================================================
namespace tatooine::color_scales {
//==============================================================================
template <floating_point Real>
struct cool_to_warm {
  using real_type  = Real;
  using this_type  = cool_to_warm<Real>;
  using color_type = vec<Real, 3>;
  static constexpr auto num_samples() -> std::size_t { return 3; }
  //==============================================================================
  std::unique_ptr<color_type[]> m_data;
  //==============================================================================
  cool_to_warm()
      : m_data{new color_type[]{{0.231373, 0.298039, 0.752941},
                                   {0.865, 0.865, 0.865},
                                   {0.705882, 0.0156863, 0.14902}}} {}
  //----------------------------------------------------------------------------
  auto data() -> auto& { return m_data; }
  auto data() const -> auto const& { return m_data; }
  //----------------------------------------------------------------------------
  auto sample(real_type const t) const {
    if (t <= 0) {
      return m_data[0];
    }
    if (t >= 1) {
      return m_data[num_samples() - 1];
    }
    t *= num_samples() - 1;
    auto const i = static_cast<size_t>(std::floor(t));
    t            = t - i;
    return m_data[i] * (1 - t) + m_data[i + 1] * t;
  }
  //----------------------------------------------------------------------------
  auto operator()(real_type const t) const { return sample(t); }
};
//==============================================================================
cool_to_warm()->cool_to_warm<real_number>;
//==============================================================================
}  // namespace tatooine::color_scales
//==============================================================================
#endif

