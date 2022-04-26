#ifndef TATOOINE_COLOR_SCALES_GNRP_H
#define TATOOINE_COLOR_SCALES_GNRP_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>

#include <memory>
//==============================================================================
namespace tatooine::color_scales {
//==============================================================================
template <floating_point Real>
struct GnRP {
  using real_type  = Real;
  using this_type  = GnRP<Real>;
  using color_type = vec<Real, 3>;
  static constexpr std::size_t num_samples() { return 17; }
  //==============================================================================
 private:
  std::unique_ptr<color_type[]> m_data;

 public:
  auto data_container() -> color_type* { return m_data; }
  auto data_container() const -> color_type const* { return m_data; }
  auto data() -> color_type* { return m_data.get(); }
  auto data() const -> color_type const* { return m_data.get(); }
  //==============================================================================
  GnRP()
      : m_data{new color_type[]{
            {0, 0.26666699999999999, 0.105882},
            {0.066435999999999995, 0.394617, 0.17477899999999999},
            {0.16885800000000001, 0.52456700000000001, 0.25767000000000001},
            {0.32387500000000002, 0.657439, 0.36101499999999997},
            {0.50488299999999997, 0.77231799999999995, 0.50634400000000002},
            {0.67843100000000001, 0.87012699999999998, 0.65490199999999998},
            {0.80392200000000003, 0.92179900000000004, 0.78039199999999997},
            {0.89711600000000002, 0.95194199999999995, 0.88281399999999999},
            {0.96739699999999995, 0.96593600000000002, 0.96747399999999995},
            {0.92802799999999996, 0.87981500000000001, 0.93056499999999998},
            {0.86605200000000004, 0.78077700000000005, 0.88289099999999998},
            {0.77500999999999998, 0.66512899999999997, 0.821376},
            {0.67566300000000001, 0.53702399999999995, 0.73702400000000001},
            {0.57847000000000004, 0.39615499999999998, 0.64598199999999995},
            {0.49234899999999998, 0.223914, 0.54755900000000002},
            {0.37554799999999999, 0.096886, 0.42329899999999998},
            {0.25097999999999998, 0, 0.29411799999999999}}} {}
  //----------------------------------------------------------------------------
  auto sample(real_type t) const {
    if (t <= 0) {
      return m_data[0];
    }
    if (t >= 1) {
      return m_data[(num_samples() - 1)];
    }
    t *= num_samples() - 1;
    auto const i = static_cast<std::size_t>(std::floor(t));
    t            = t - i;
    return m_data[i] * (1 - t) + m_data[i + 1] * t;
  }
  auto operator()(real_type const t) const { return sample(t); }
};
//==============================================================================
GnRP()->GnRP<real_number>;
//==============================================================================
}  // namespace tatooine::color_scales
//==============================================================================
#endif

