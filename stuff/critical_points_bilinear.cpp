#include <array>
#include <cmath>
#include <iostream>
#include <vector>

constexpr auto eval_bilinear(const double u00, const double u10, const double u01, const double u11, const double v00,
                             const double v10, const double v01, const double v11, const double s, const double t) {
  return std::array{
      (1 - s) * (1 - t) * u00 + s * (1 - t) * u10 + (1 - s) * t * u01 + s * t * u11,
      (1 - s) * (1 - t) * v00 + s * (1 - t) * v10 + (1 - s) * t * v01 + s * t * v11,
  };
}

auto solve_bilinear(const double u00, const double u10, const double u01, const double u11, const double v00,
                    const double v10, const double v01, const double v11) {
  const double a = u01 * v10;
  const double b = u10 * v01;
  const double c = 2 * u00;
  const double d = 2 * u01;
  const double e = 2 * u10;
  const double f = 2 * u11;
  const double g = sqrt(u00 * u00 * v11 * v11 + (-c * a - c * b + (4 * u01 * u10 - c * u11) * v00) * v11 +
                        u01 * a * v10 + ((4 * u00 * u11 - d * u10) * v01 - d * u11 * v00) * v10 + u10 * b * v01 -
                        e * u11 * v00 * v01 + u11 * u11 * v00 * v00);
  const double h = u00 * v11;
  const double i = 1 / ((e - c) * v11 + (d - f) * v10 + (c - e) * v01 + (f - d) * v00);
  const double j = 1 / ((d - c) * v11 + (c - d) * v10 + (e - f) * v01 + (f - e) * v00);

  const double s0 = -(g + h - a + (u10 - c) * v01 + (d - u11) * v00) * i;
  const double s1 = (g - h + a + (c - u10) * v01 + (u11 - d) * v00) * i;
  const double t0 = -(g + h + (u01 - c) * v10 - b + (e - u11) * v00) * j;
  const double t1 = (g - h + (c - u01) * v10 + b + (u11 - e) * v00) * j;

  std::vector<std::array<double, 2>> solutions;
  if (s0 > -1e-7 && s0 < 1 + 1e-7) {
    if (t0 > -1e-7 && t0 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(u00, u10, u01, u11, v00, v10, v01, v11, s0, t0);
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s0, t0});
      }
    }
    if (t1 > -1e-7 && t1 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(u00, u10, u01, u11, v00, v10, v01, v11, s0, t1);
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s0, t1});
      }
    }
  }
  if (s1 > -1e-7 && s1 < 1 + 1e-7) {
    if (t0 > -1e-7 && t0 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(u00, u10, u01, u11, v00, v10, v01, v11, s1, t0);
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s1, t0});
      }
    }
    if (t1 > -1e-7 && t1 < 1 + 1e-7) {
      if (auto [x, y] = eval_bilinear(u00, u10, u01, u11, v00, v10, v01, v11, s1, t1);
          std::abs(x) < 1e-7 && std::abs(y) < 1e-7) {
        solutions.push_back({s1, t1});
      }
    }
  }
  return solutions;
}

int main() {
  double u00 = 3.0 / 4.0, u10 = -9.0 / 4.0, u01 = -1.0 / 4.0, u11 = 3.0 / 4.0;
  double v00 = 3.0 / 4.0, v10 = -1.0 / 4.0, v01 = -9.0 / 4.0, v11 = 3.0 / 4.0;
  auto   sols = solve_bilinear(u00, u10, u01, u11, v00, v10, v01, v11);
  for (const auto [s, t] : sols) {
    auto [u, v] = eval_bilinear(u00, u10, u01, u11, v00, v10, v01, v11, s, t);
    std::cerr << "[" << s << ", " << t << "] = [" << u << ", " << v << "]\n";
  }
}
