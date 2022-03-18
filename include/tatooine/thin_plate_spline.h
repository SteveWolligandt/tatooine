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
auto constexpr thin_plate_spline_diff2 =
    [](auto const x) -> decltype(x) {
  return 2 * gcem::log(x) + 3;
};
//------------------------------------------------------------------------------
template <std::size_t N=1>
auto constexpr diff(decltype(thin_plate_spline) const& /*f*/) {
  if constexpr (N == 0) {
    return thin_plate_spline;
  } else if constexpr (N == 1) {
    return thin_plate_spline_diff1;
  } else if constexpr (N == 2) {
    return thin_plate_spline_diff2;
  }
}
//------------------------------------------------------------------------------
template <std::size_t N=1>
auto constexpr diff(decltype(thin_plate_spline_diff1) const& /*f*/) {
  if constexpr (N == 0) {
    return thin_plate_spline_diff1;
  } else if constexpr (N == 1) {
    return thin_plate_spline_diff2;
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
