#ifndef TATOOINE_DETAIL_POINTSET_MOVING_LEAST_SQUARES_SAMPLERN_H
#define TATOOINE_DETAIL_POINTSET_MOVING_LEAST_SQUARES_SAMPLERN_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/pointset.h>
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
#if TATOOINE_FLANN_AVAILABLE
template <floating_point Real, std::size_t NumDimensions, typename ValueType,
          invocable<Real> F>
struct moving_least_squares_sampler;
#endif
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
