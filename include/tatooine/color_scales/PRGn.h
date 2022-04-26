#ifndef TATOOINE_COLOR_SCALES_PRGN_H
#define TATOOINE_COLOR_SCALES_PRGN_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>

#include <memory>
//==============================================================================
namespace tatooine::color_scales {
//==============================================================================
template <floating_point Real>
struct PRGn {
  using real_type  = Real;
  using this_type  = PRGn<Real>;
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
  PRGn()
      : m_data{new color_type[]{
            {0.25097999999999998, 0, 0.29411799999999999},
            {0.38385200000000003, 0.10334500000000001, 0.43191099999999999},
            {0.49773200000000001, 0.234679, 0.55371000000000004},
            {0.58385200000000004, 0.40692, 0.65213399999999999},
            {0.68196800000000002, 0.54517499999999997, 0.74256100000000003},
            {0.78069999999999995, 0.67235699999999998, 0.82522099999999998},
            {0.87174200000000002, 0.78800499999999996, 0.88673599999999997},
            {0.93048799999999998, 0.88519800000000004, 0.93287200000000003},
            {0.96632099999999999, 0.96808899999999998, 0.96585900000000002},
            {0.89250300000000005, 0.95086499999999996, 0.877278},
            {0.79607799999999995, 0.91857, 0.77254900000000004},
            {0.67058799999999996, 0.86689700000000003, 0.64705900000000005},
            {0.49319499999999999, 0.76539800000000002, 0.49665500000000001},
            {0.31418699999999999, 0.64913500000000002, 0.35455599999999998},
            {0.15917000000000001, 0.51626300000000003, 0.25121100000000002},
            {0.062283999999999999, 0.38662099999999999, 0.17047300000000001},
            {0, 0.26666699999999999, 0.105882}}} {}
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
PRGn()->PRGn<real_number>;
//==============================================================================
}  // namespace tatooine::color_scales
//==============================================================================
#endif

