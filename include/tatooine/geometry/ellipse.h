#ifndef TATOOINE_GEOMETRY_ELLIPSOID_H
#define TATOOINE_GEOMETRY_ELLIPSOID_H
//==============================================================================
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/linspace.h>
#include <tatooine/real.h>

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <floating_point Real>
using ellipse = hyper_ellipse<Real, 2>;
//template <floating_point Real>
//struct ellipse : hyper_ellipse<Real, 2> {
//  using this_t   = ellipse;
//  using parent_type = hyper_ellipse<Real, 2>;
//  using parent_type::parent_type;
//};
////==============================================================================
//ellipse()->ellipse<real_t>;
////------------------------------------------------------------------------------
//template <floating_point Real>
//ellipse(Real const) -> ellipse<Real>;
////------------------------------------------------------------------------------
//template <floating_point Real0, floating_point Real1>
//ellipse(Real0 const, Real1 const) -> ellipse<common_type<Real0, Real1>>;
////------------------------------------------------------------------------------
//template <floating_point Real0, floating_point Real1, floating_point Real2>
//ellipse(vec<Real0, 2> const&, Real1 const, Real2 const)
//    -> ellipse<common_type<Real0, Real1, Real2>>;
////------------------------------------------------------------------------------
//template <floating_point Real0, floating_point Real1>
//ellipse(vec<Real0, 2> const&, vec<Real1, 2> const&)
//    -> ellipse<common_type<Real0, Real1>>;
////==============================================================================
}  // namespace tatooine::geometry
////==============================================================================
//namespace tatooine::reflection {
////==============================================================================
//template <typename Real>
//TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
//    (geometry::ellipse<Real>),
//    TATOOINE_REFLECTION_INSERT_METHOD(center, center()),
//    TATOOINE_REFLECTION_INSERT_METHOD(S, S()))
////==============================================================================
//}  // namespace tatooine::reflection
////==============================================================================
#endif
