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
  using namespace boost;
  using namespace adaptors;
  linspace<Real> radial{0.0, M_PI * 2, num_vertices};
  radial.pop_back();

  line<Real, 2> discretization;
  auto          radian_to_cartesian = [](auto const t) {
    return vec{std::cos(t), std::sin(t)};
  };
  auto out_it = std::back_inserter(discretization);
  copy(radial | transformed(radian_to_cartesian), out_it);
  discretization.set_closed(true);
  for (auto const v : discretization.vertices()) {
    discretization[v] = e.S() * discretization[v] + e.center();
  }
  return discretization;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
