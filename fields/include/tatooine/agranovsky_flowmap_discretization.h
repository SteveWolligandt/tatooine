#ifndef TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/regular_flowmap_discretization.h>
#include <tatooine/staggered_flowmap_discretization.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions>
using agranovsky_flowmap_discretization = staggered_flowmap_discretization<
    regular_flowmap_discretization<Real, NumDimensions>>;
template <typename Real>
using AgranovskyFlowmapDiscretization2 =
    agranovsky_flowmap_discretization<Real, 2>;
template <typename Real>
using AgranovskyFlowmapDiscretization3 =
    agranovsky_flowmap_discretization<Real, 3>;
template <std::size_t NumDimensions>
using AgranovskyFlowmapDiscretization =
    agranovsky_flowmap_discretization<real_number, NumDimensions>;
using agranovsky_flowmap_discretization2 =
    AgranovskyFlowmapDiscretization<2>;
using agranovsky_flowmap_discretization3 =
    AgranovskyFlowmapDiscretization<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
