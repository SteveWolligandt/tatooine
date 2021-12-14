#ifndef TATOOINE_RECTILINEAR_GRID_DIMENSION_H
#define TATOOINE_RECTILINEAR_GRID_DIMENSION_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
concept rectilinear_grid_dimension = floating_point_range<T>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
