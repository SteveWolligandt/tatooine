#ifndef TATOOINE_COLOR_SCALES_BURD_H
#define TATOOINE_COLOR_SCALES_BURD_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>

#include <memory>
//==============================================================================
namespace tatooine::color_scales {
//==============================================================================
template <floating_point Real>
struct BuRD {
  using real_type  = Real;
  using this_type  = BuRD<Real>;
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
  BuRD()
      : m_data{new color_type[]{
            {0.019608, 0.18823500000000001, 0.38039200000000001},
            {0.088503999999999999, 0.32110699999999998, 0.56493700000000002},
            {0.16339899999999999, 0.44498300000000002, 0.69750100000000004},
            {0.247059, 0.55570900000000001, 0.75409499999999996},
            {0.420684, 0.67643200000000003, 0.818685},
            {0.60645899999999997, 0.78977299999999995, 0.88027699999999998},
            {0.76147600000000004, 0.86851199999999995, 0.92456700000000003},
            {0.87804700000000002, 0.92572100000000002, 0.95194199999999995},
            {0.96908899999999998, 0.96647400000000006, 0.96493700000000004},
            {0.98385199999999995, 0.89757799999999999, 0.84682800000000003},
            {0.98246800000000001, 0.80069199999999996, 0.70611299999999999},
            {0.96032300000000004, 0.66781999999999997, 0.53633200000000003},
            {0.89457900000000001, 0.50380599999999998, 0.39976899999999999},
            {0.81706999999999996, 0.33217999999999998, 0.28104600000000002},
            {0.72848900000000005, 0.15501699999999999, 0.19738600000000001},
            {0.576932, 0.055363000000000002, 0.14924999999999999},
            {0.403922, 0, 0.121569}}} {}
  //----------------------------------------------------------------------------
  auto sample(real_type const t) const {
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
BuRD()->BuRD<real_number>;
//==============================================================================
}  // namespace tatooine::color_scales
//==============================================================================
#endif

