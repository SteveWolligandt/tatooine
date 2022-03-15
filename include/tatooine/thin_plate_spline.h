#ifndef TATOOINE_THIN_PLATE_SPLINE_H
#define TATOOINE_THIN_PLATE_SPLINE_H
//==============================================================================
namespace tatooine {
//==============================================================================
auto constexpr thin_plate_spline =
    [](auto const sqr_dist) -> decltype(sqr_dist) {
  if (sqr_dist == 0) {
    return 0;
  }
  return sqr_dist * gcem::log(sqr_dist) / 2;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
