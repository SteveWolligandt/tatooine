#ifndef TATOOINE_GEOMETRY_ELLIPSOID_H
#define TATOOINE_GEOMETRY_ELLIPSOID_H
//==============================================================================
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <floating_point Real>
struct ellipsoid : hyper_ellipse<Real, 3> {
  using this_t   = ellipsoid;
  using parent_t = hyper_ellipse<Real, 3>;
  using parent_t::parent_t;
};
//==============================================================================
ellipsoid()->ellipsoid<real_t>;
//------------------------------------------------------------------------------
template <typename Real0, typename Real1, typename Real2>
ellipsoid(Real0 const, Real1 const, Real2 const)
    -> ellipsoid<common_type<Real0, Real1, Real2>>;
//------------------------------------------------------------------------------
template <typename Real0, typename Real1, typename Real2>
ellipsoid(vec<Real0, 3> const&, vec<Real1, 3> const&, vec<Real2, 3> const&)
    -> ellipsoid<common_type<Real0, Real1, Real2>>;
//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================
namespace tatooine::reflection {
//==============================================================================
template <typename Real>
TATOOINE_MAKE_TEMPLATED_ADT_REFLECTABLE(
    (geometry::ellipsoid<Real>),
    TATOOINE_REFLECTION_INSERT_METHOD(center, center()),
    TATOOINE_REFLECTION_INSERT_METHOD(S, S()))
//==============================================================================
}  // namespace tatooine::reflection
//==============================================================================
#endif
