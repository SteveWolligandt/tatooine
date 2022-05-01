#ifndef TATOOINE_DETAIL_POINTSET_MOVING_LEAST_SQUARES_SAMPLERN_H
#define TATOOINE_DETAIL_POINTSET_MOVING_LEAST_SQUARES_SAMPLERN_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/pointset.h>
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <floating_point Real, std::size_t N, typename T, invocable<Real> F>
struct moving_least_squares_sampler;
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
