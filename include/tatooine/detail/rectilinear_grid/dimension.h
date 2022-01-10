#ifndef TATOOINE_DETAIL_RECTILINEAR_GRID_DIMENSION_H
#define TATOOINE_DETAIL_RECTILINEAR_GRID_DIMENSION_H
//==============================================================================
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
//==============================================================================
template <typename T>
concept dimension = floating_point_range<T>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
