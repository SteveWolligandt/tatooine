//==============================================================================
#include <tatooine/mpi/feeders/analytical_function.h>

#include <cmath>
#include <stdexcept>
#include <tuple>
//==============================================================================
namespace tatooine::mpi::feeders {
//==============================================================================
EggBox::EggBox(double a, double bx, double by, double cx, double cy)
    : a{a}, bx{bx}, by{by}, cx{cx}, cy{cy} {}
//------------------------------------------------------------------------------
double EggBox::s(double x, double y, double z, double /* t */) const {
  // TODO vary parameters with t
  return a * sin(bx * (x + cx)) * cos(by * (y + cy)) - z;
}
//------------------------------------------------------------------------------
vec3 EggBox::g(double x, double y, double /* z */, double /* t */) const {
  // TODO vary parameters with t
  return {a * bx * cos(bx * (cx + x)) * cos(by * (cy + y)),
          -a * by * sin(bx * (cx + x)) * sin(by * (cy + y)), -1};
}
//------------------------------------------------------------------------------
vec3 EggBox::v(double /* x */, double /* y */, double /* z */, double /* t */) const {
  return vec3::zeros();
}
//------------------------------------------------------------------------------
TileBox::TileBox(double k, double l)
    : ScalarFieldInFlow(bbox_t{0, 1, 0, 1, -1, 1, 0, 1}), k{k}, l{l} {}
//------------------------------------------------------------------------------
double TileBox::s(double x, double y, double z, double t) const {
  x = x - floor(x - l * t) - l * t;
  if (std::sqrt(std::pow((x - 0.5), 2) + std::pow((z - 0), 2)) < 0.5 && y > 0 &&
      y < 1) {
    return -std::sin(
               k * t *
               std::pow(std::sin(M_PI * (0.5 * std::sqrt(4.0 * x * x - 4.0 * x +
                                                         1 + 4.0 * z * z) +
                                         0.5)),
                        2) *
               std::pow(std::sin(M_PI * y), 2)) *
               (x - 0.5) +
           std::cos(k * t *
                    std::pow(std::sin(M_PI *
                                      ((0.5) * std::sqrt(4.0 * x * x - 4.0 * x +
                                                         1 + 4.0 * z * z) +
                                       0.5)),
                             2) *
                    std::pow(std::sin(M_PI * y), 2)) *
               z;
  } else {
    return z;
  }
}
//------------------------------------------------------------------------------
vec3 TileBox::g(double x, double y, double z, double t) const {
  x = x - std::floor(x - l * t) - l * t;
  if (std::sqrt(std::pow((x - 0.5), 2) + std::pow((z - 0), 2)) < 0.5 && y > 0 && y < 1) {
    auto A = std::sqrt(4.0 * x * x - 4.0 * x + 4.0 * z * z + 1);
    auto B = 0.5 * M_PI + 0.5 * M_PI * A;
    auto C = k * t * std::pow(std::sin(B), 2) * std::pow(std::sin(M_PI * y), 2);

    if (A == 0) {
      return vec3::zeros();
    }

    auto sX = -M_PI * std::pow((1 - 2 * x), 2) * k * t * std::cos(B) *
                  std::cos(C) * std::sin(B) * std::pow(std::sin(M_PI * y), 2) /
                  A +
              2 * M_PI * (1 - 2 * x) * k * t * z * std::cos(B) * std::sin(B) *
                  std::sin(C) * pow(sin(M_PI * y), 2) / A -
              sin(C);

    auto sY = M_PI * (1 - 2 * x) * k * t * std::cos(C) * std::cos(M_PI * y) *
                  std::pow(std::sin(B), 2) * std::sin(M_PI * y) -
              2 * M_PI * k * t * z * std::cos(M_PI * y) *
                  std::pow(std::sin(B), 2) * std::sin(C) * std::sin(M_PI * y);

    auto sZ = 2 * M_PI * (1 - 2 * x) * k * t * z * std::cos(B) * std::cos(C) *
                  std::sin(B) * std::pow(std::sin(M_PI * y), 2) / A -
              4 * M_PI * k * t * z * z * std::cos(B) * std::sin(B) *
                  std::sin(C) * std::pow(std::sin(M_PI * y), 2) / A +
              std::cos(C);
    return {sX, sY, sZ};
  } else {
    return vec3{0, 0, 1};
  }
}
//------------------------------------------------------------------------------
vec3 TileBox::v(double /* x */, double /* y */, double /* z */,
                double /* t */) const {
  return vec3{1, 0, 0} * l;
}
//------------------------------------------------------------------------------
Plane::Plane() : ScalarFieldInFlow(bbox_t{0, 1, 0, 1, -1, 1, 0, 1}) {}
//------------------------------------------------------------------------------
double Plane::s(double /*x*/, double /*y*/, double z, double /*t*/) const {
  return z;
}
//------------------------------------------------------------------------------
vec3 Plane::g(double /*x*/, double /*y*/, double /*z*/, double /*t*/) const {
  return vec3{0, 0, 1};
}
//------------------------------------------------------------------------------
vec3 Plane::v(double x, double /*y*/, double /*z*/, double /*t*/) const {
  return vec3{1, 0, 0} * std::exp(x) - vec3{1, 0, 0};
}
//------------------------------------------------------------------------------
std::tuple<double, double, double> interpSteps(double t, std::set<double> tsteps) {
  auto f = tsteps.upper_bound(t);
  if (f == tsteps.begin() || f == tsteps.end()) {
    throw std::out_of_range(
        "The supplied time is not in between the "
        "available time steps");
  }

  auto high_t = *f;
  auto low_t  = *(--f);
  auto w      = (t - low_t) / (high_t - low_t);
  return std::tuple{low_t, high_t, w};
}
//------------------------------------------------------------------------------
InterpolatedField::InterpolatedField(std::unique_ptr<ScalarFieldInFlow>&& base,
                                     std::set<double>                     steps)
    : _base{std::move(base)}, _steps{steps} {}
//------------------------------------------------------------------------------
double InterpolatedField::s(double x, double y, double z, double t) const {
  auto const [low_t, high_t, w] = interpSteps(t, _steps);
  return _base->s(x, y, z, low_t) * (1 - w) + _base->s(x, y, z, high_t) * w;
}
//------------------------------------------------------------------------------
vec3 InterpolatedField::g(double x, double y, double z, double t) const {
  auto const [low_t, high_t, w] = interpSteps(t, _steps);
  return _base->g(x, y, z, low_t) * (1 - w) + _base->g(x, y, z, high_t) * w;
}
//------------------------------------------------------------------------------
vec3 InterpolatedField::v(double x, double y, double z, double t) const {
  auto const [low_t, high_t, w] = interpSteps(t, _steps);
  return _base->v(x, y, z, low_t) * (1 - w) + _base->v(x, y, z, high_t) * w;
}
//==============================================================================
}  // namespace tatooine::mpi::feeders
//==============================================================================
