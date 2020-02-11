#include <cmath>
#include <vector>

#include "tensor.h"

//==============================================================================
namespace tatooine {
namespace detail {
//==============================================================================
template <typename Real, typename T0, typename T1, typename T2, typename T3>
constexpr auto eval_bilinear(const base_tensor<T0, Real, 2>& v00,
                             const base_tensor<T1, Real, 2>& v10,
                             const base_tensor<T2, Real, 2>& v01,
                             const base_tensor<T3, Real, 2>& v11,
                             const Real s, const Real t) {
  return vec<Real, 2>{
      (1 - s) * (1 - t) * v00(0) + s * (1 - t) * v10(0) + (1 - s) * t * v01(0) +
          s * t * v11(0),
      (1 - s) * (1 - t) * v00(1) + s * (1 - t) * v10(1) + (1 - s) * t * v01(1) +
          s * t * v11(1),
  };
}
//------------------------------------------------------------------------------
template <typename Real, typename T0, typename T1, typename T2, typename T3>
auto solve_bilinear(const base_tensor<T0, Real, 2>& v00,
                    const base_tensor<T1, Real, 2>& v10,
                    const base_tensor<T2, Real, 2>& v01,
                    const base_tensor<T3, Real, 2>& v11) {
  const Real a = v01(0) * v10(1);
  const Real b = v10(0) * v01(1);
  const Real c = 2 * v00(0);
  const Real d = 2 * v01(0);
  const Real e = 2 * v10(0);
  const Real f = 2 * v11(0);
  const Real g = sqrt(
      v00(0) * v00(0) * v11(1) * v11(1) +
      (-c * a - c * b + (4 * v01(0) * v10(0) - c * v11(0)) * v00(1)) * v11(1) +
      v01(0) * a * v10(1) +
      ((4 * v00(0) * v11(0) - d * v10(0)) * v01(1) - d * v11(0) * v00(1)) *
          v10(1) +
      v10(0) * b * v01(1) - e * v11(0) * v00(1) * v01(1) +
      v11(0) * v11(0) * v00(1) * v00(1));
  const Real h = v00(0) * v11(1);
  const Real i = 1 / ((e - c) * v11(1) + (d - f) * v10(1) + (c - e) * v01(1) +
                        (f - d) * v00(1));
  const Real j = 1 / ((d - c) * v11(1) + (c - d) * v10(1) + (e - f) * v01(1) +
                        (f - e) * v00(1));

  const Real s0 =
      -(g + h - a + (v10(0) - c) * v01(1) + (d - v11(0)) * v00(1)) * i;
  const Real s1 =
      (g - h + a + (c - v10(0)) * v01(1) + (v11(0) - d) * v00(1)) * i;
  const Real t0 =
      -(g + h + (v01(0) - c) * v10(1) - b + (e - v11(0)) * v00(1)) * j;
  const Real t1 =
      (g - h + (c - v01(0)) * v10(1) + b + (v11(0) - e) * v00(1)) * j;

  std::vector<vec<Real, 2>> solutions;
  if (s0 > -1e-7 && s0 < 1 + 1e-7) {
    if (t0 > -1e-7 && t0 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(v00, v10, v01, v11, s0, t0).data();
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s0, t0});
      }
    }
    if (t1 > -1e-7 && t1 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(v00, v10, v01, v11, s0, t1).data();
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s0, t1});
      }
    }
  }
  if (s1 > -1e-7 && s1 < 1 + 1e-7) {
    if (t0 > -1e-7 && t0 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(v00, v10, v01, v11, s1, t0).data();
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s1, t0});
      }
    }
    if (t1 > -1e-7 && t1 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(v00, v10, v01, v11, s1, t1).data();
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
