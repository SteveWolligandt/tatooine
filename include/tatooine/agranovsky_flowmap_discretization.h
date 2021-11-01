#ifndef TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/regular_flowmap_discretization.h>
#include <tatooine/staggered_flowmap_discretization.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
using agranovsky_flowmap_discretization =
    staggered_flowmap_discretization<regular_flowmap_discretization<Real, N>>;
template <size_t N>
using AgranovskyFlowmapDiscretization =
    agranovsky_flowmap_discretization<real_t, N>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
