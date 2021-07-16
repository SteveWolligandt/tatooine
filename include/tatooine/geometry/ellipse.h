#ifndef TATOOINE_GEOMETRY_ELLIPSOID_H
#define TATOOINE_GEOMETRY_ELLIPSOID_H
//==============================================================================
#include <tatooine/geometry/hyper_ellipse.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <floating_point Real>
struct ellipse : hyper_ellipse<Real, 2> {
  using this_t   = ellipse;
  using parent_t = hyper_ellipse<Real, 2>;
  using parent_t::parent_t;
};
//==============================================================================
ellipse()->ellipse<real_t>;
//------------------------------------------------------------------------------
template <typename Real0, typename Real1>
ellipse(Real0 const, Real1 const) -> ellipse<common_type<Real0, Real1>>;
//------------------------------------------------------------------------------
template <typename Real0, typename Real1>
ellipse(vec<Real0, 3> const&, vec<Real1, 3> const&)
    -> ellipse<common_type<Real0, Real1>>;

//==============================================================================
template <typename Real>
auto discretize(hyper_ellipse<Real, 2> const& e, size_t const num_vertices) {
  auto d = discretize(sphere<Real, 2>{}, num_vertices);
  for (auto const v : d.vertices()) {
    d[v] = e.S() * d[v];
  }
  return d;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
