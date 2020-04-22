#ifndef TATOOINE_STEADIFICATION_PATHSURFACE_H
#define TATOOINE_STEADIFICATION_PATHSURFACE_H
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/interpolation.h>
#include <tatooine/random.h>
#include <tatooine/streamsurface.h>
//==============================================================================
namespace tatooine::steadification {
//==============================================================================
template <typename V, typename Real>
auto pathsurface(
    const vectorfield<V, Real, 2>&                            v,
    const parameterized_line<Real, 2, interpolation::linear>& seedcurve,
    Real u0t0, Real u1t0, Real btau, Real ftau, size_t seed_res,
    Real stepsize) {
  using namespace VC::odeint;
  integration::vclibs::rungekutta43<Real, 2, interpolation::hermite> integrator{
      integration::vclibs::abs_tol = 1e-6, integration::vclibs::rel_tol = 1e-6,
      integration::vclibs::initial_step = 0,
      integration::vclibs::max_step     = 0.1};
  streamsurface surf{v, u0t0, u1t0, seedcurve, integrator};

  if (u0t0 == u1t0) {
    // if (seedcurve.vertex_at(0)(0) != seedcurve.vertex_at(1)(0) &&
    //    seedcurve.vertex_at(0)(1) != seedcurve.vertex_at(1)(1)) {
    simple_tri_mesh<Real, 2> mesh =
        surf.discretize(seed_res, stepsize, btau, ftau);
    auto& uvprop   = mesh.template add_vertex_property<vec2>("uv");
    auto& vprop    = mesh.template add_vertex_property<vec2>("v");
    auto& curvprop = mesh.template add_vertex_property<Real>("curvature");

    for (auto vertex : mesh.vertices()) {
      const auto& uv             = uvprop[vertex];
      const auto& integral_curve = surf.streamline_at(uv(0), 0, 0);
      curvprop[vertex]           = integral_curve.curvature(uv(1));
      if (std::isnan(curvprop[vertex])) {
        std::cerr << "got nan!\n";
        std::cerr << "t0     = " << u0t0 << '\n';
        std::cerr << "pos     = " << integral_curve(uv(1)) << '\n';
        std::cerr << "v       = " << v(integral_curve(uv(1)), uv(1)) << '\n';
        std::cerr << "tau     = " << uv(1) << '\n';
        std::cerr << "tangent = " << integral_curve.tangent(uv(1)) << '\n';
      }
      if (v.in_domain(mesh[vertex], uv(1))) {
        vprop[vertex] =
            v(vec{mesh[vertex](0), mesh[vertex](1)}, uvprop[vertex](1));
      } else {
        vprop[vertex] = vec<Real, 2>{0.0 / 0.0, 0.0 / 0.0};
      }
    }
    return std::pair{std::move(mesh), std::move(surf)};
  } else {
    return std::pair{simple_tri_mesh<Real, 2>{}, std::move(surf)};
  }
}
//----------------------------------------------------------------------------
template <typename V, typename Real>
auto pathsurface(const vectorfield<V, Real, 2>& v,
                 const grid_edge<Real, 3>& edge, Real btau, Real ftau,
                 size_t seed_res, Real stepsize) {
  const auto        v0 = edge.first.position();
  const auto        v1 = edge.second.position();
  const parameterized_line<Real, 2, interpolation::linear> seedcurve{
      {vec{v0(0), v0(1)}, 0}, {vec{v1(0), v1(1)}, 1}};
  return pathsurface(v, seedcurve, v0(2), v1(2), btau, ftau, seed_res,
                     stepsize);
}
//==============================================================================
}  // namespace tatooine::steadification
//==============================================================================
#endif
