#ifndef TATOOINE_THIN_PLATE_SPLINE_H
#define TATOOINE_THIN_PLATE_SPLINE_H
//==============================================================================
#include <tatooine/math.h>
//==============================================================================
namespace tatooine {
//==============================================================================
auto constexpr thin_plate_spline = [](auto const x) -> decltype(x) {
  if (x == 0) {
    return 0;
  }
  return x * x * gcem::log(x);
};
//------------------------------------------------------------------------------
auto constexpr thin_plate_spline_from_squared =
    [](auto const squared_x) -> decltype(squared_x) {
  if (squared_x == 0) {
    return 0;
  }
  return squared_x * gcem::log(squared_x) / 2;
};
//------------------------------------------------------------------------------
auto constexpr thin_plate_spline_diff1 =
    [](auto const x) -> decltype(x) {
  return 2 * x * gcem::log(x) + x;
};
//------------------------------------------------------------------------------
auto constexpr diff(decltype(thin_plate_spline) const& /*f*/) {
  return thin_plate_spline_diff1;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
