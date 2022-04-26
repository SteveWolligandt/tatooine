#ifndef TATOOINE_COLOR_SCALES_GYPI_H
#define TATOOINE_COLOR_SCALES_GYPI_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>

#include <memory>
//==============================================================================
namespace tatooine::color_scales {
//==============================================================================
template <floating_point Real>
struct GYPi {
  using real_type  = Real;
  using this_type  = GYPi<Real>;
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
  GYPi()
      : m_data{new color_type[]{
            {0.15294099999999999, 0.39215699999999998, 0.098039000000000001},
            {0.246444, 0.50534400000000002, 0.117724},
            {0.35194199999999998, 0.614533, 0.16139899999999999},
            {0.47497099999999998, 0.71787800000000002, 0.24013799999999999},
            {0.61199499999999996, 0.811226, 0.392849},
            {0.74632799999999999, 0.89311799999999997, 0.56532099999999996},
            {0.85951599999999995, 0.94233, 0.74740499999999999},
            {0.92810499999999996, 0.96386000000000005, 0.87566299999999997},
            {0.96908899999999998, 0.96685900000000002, 0.96801199999999998},
            {0.98385199999999995, 0.91026499999999999, 0.94832799999999995},
            {0.97923899999999997, 0.83321800000000001, 0.91464800000000002},
            {0.949712, 0.72987299999999999, 0.86297599999999997},
            {0.90565200000000001, 0.58292999999999995, 0.76355200000000001},
            {0.85521000000000003, 0.41007300000000002, 0.65221099999999999},
            {0.79369500000000004, 0.183699, 0.53164199999999995},
            {0.68373700000000004, 0.063898999999999997, 0.420761},
            {0.556863, 0.0039220000000000001, 0.32156899999999999}}} {}
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
GYPi()->GYPi<real_number>;
//==============================================================================
}  // namespace tatooine::color_scales
//==============================================================================
#endif

