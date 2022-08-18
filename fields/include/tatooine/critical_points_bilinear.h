#include <tatooine/tensor.h>

#include <cmath>
#include <vector>
//==============================================================================
namespace tatooine {
namespace detail {
//==============================================================================
template <typename Real, typename T0, typename T1, typename T2, typename T3>
constexpr auto eval_bilinear(base_tensor<T0, Real, 2> const& v00,
                             base_tensor<T1, Real, 2> const& v10,
                             base_tensor<T2, Real, 2> const& v01,
                             base_tensor<T3, Real, 2> const& v11, Real const s,
                             Real const t) {
  return vec<Real, 2>{
      (1 - s) * (1 - t) * v00(0) + s * (1 - t) * v10(0) + (1 - s) * t * v01(0) +
          s * t * v11(0),
      (1 - s) * (1 - t) * v00(1) + s * (1 - t) * v10(1) + (1 - s) * t * v01(1) +
          s * t * v11(1),
  };
}
//------------------------------------------------------------------------------
template <typename Real, typename T0, typename T1, typename T2, typename T3>
auto solve_bilinear(base_tensor<T0, Real, 2> const& v00,
                    base_tensor<T1, Real, 2> const& v10,
                    base_tensor<T2, Real, 2> const& v01,
                    base_tensor<T3, Real, 2> const& v11) {
  auto const a = v01(0) * v10(1);
  auto const b = v10(0) * v01(1);
  auto const c = 2 * v00(0);
  auto const d = 2 * v01(0);
  auto const e = 2 * v10(0);
  auto const f = 2 * v11(0);
  auto const g = std::sqrt(
      v00(0) * v00(0) * v11(1) * v11(1) +
      (-c * a - c * b + (4 * v01(0) * v10(0) - c * v11(0)) * v00(1)) * v11(1) +
      v01(0) * a * v10(1) +
      ((4 * v00(0) * v11(0) - d * v10(0)) * v01(1) - d * v11(0) * v00(1)) *
          v10(1) +
      v10(0) * b * v01(1) - e * v11(0) * v00(1) * v01(1) +
      v11(0) * v11(0) * v00(1) * v00(1));
  auto const h = v00(0) * v11(1);
  auto const i = 1 / ((e - c) * v11(1) + (d - f) * v10(1) + (c - e) * v01(1) +
                      (f - d) * v00(1));
  auto const j = 1 / ((d - c) * v11(1) + (c - d) * v10(1) + (e - f) * v01(1) +
                      (f - e) * v00(1));

  auto const s0 =
      -(g + h - a + (v10(0) - c) * v01(1) + (d - v11(0)) * v00(1)) * i;
  auto const s1 =
      (g - h + a + (c - v10(0)) * v01(1) + (v11(0) - d) * v00(1)) * i;
  auto const t0 =
      -(g + h + (v01(0) - c) * v10(1) - b + (e - v11(0)) * v00(1)) * j;
  auto const t1 =
      (g - h + (c - v01(0)) * v10(1) + b + (v11(0) - e) * v00(1)) * j;

  std::vector<vec<Real, 2>> solutions;
  if (s0 > -1e-7 && s0 < 1 + 1e-7) {
    if (t0 > -1e-7 && t0 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(v00, v10, v01, v11, s0, t0).internal_container();
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s0, t0});
      }
    }
    if (t1 > -1e-7 && t1 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(v00, v10, v01, v11, s0, t1).internal_container();
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s0, t1});
      }
    }
  }
  if (s1 > -1e-7 && s1 < 1 + 1e-7) {
    if (t0 > -1e-7 && t0 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(v00, v10, v01, v11, s1, t0).internal_container();
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s1, t0});
      }
    }
    if (t1 > -1e-7 && t1 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(v00, v10, v01, v11, s1, t1).internal_container();
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s1, t1});
      }
    }
  }
  return solutions;
}
//==============================================================================
}  // namespace detail
}  // namespace tatooine
//==============================================================================
