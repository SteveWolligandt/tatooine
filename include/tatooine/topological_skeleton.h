#ifndef TATOOINE_TOPOLIGICAL_SKELEKTON
#define TATOOINE_TOPOLIGICAL_SKELEKTON

#include <vector>
#include "tensor.h"
#include "grid_sampler.h"
#include "sampled_field.h"
#include "integration/integrator.h"
#include "line.h"
#include "critical_points.h"
#include "boundary_switch.h"

//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N, template <typename> typename Interpolator>
struct topological_skeleton;
//==============================================================================
// 2D
//==============================================================================
template <typename Real, template <typename> typename Interpolator>
struct topological_skeleton<Real, 2, Interpolator> {
  std::vector<vec<Real, 2>> saddles, centers, sources, sinks, repelling_foci,
      attracting_foci, boundary_switch_points;
  std::vector<parameterized_line<Real, 2, Interpolator>> separatrices;
};
//==============================================================================
/// computes the topological skeleton of a two-dimensional piecewise bilinear
/// vectorfield
template <typename Real, typename Integrator,
          template <typename> typename Interpolator>
auto compute_topological_skeleton(
    const sampled_field<
        grid_sampler<Real, 2, vec<Real, 2>, interpolation::linear,
                     interpolation::linear>,
        Real, 2, 2>& v,
    const integration::integrator<Real, 2, Interpolator, Integrator>&
        integrator, const Real tau = 100, const Real  eps = 1e-7) {
  using integral_t = typename Integrator::integral_t;
  topological_skeleton<Real, 2, Interpolator> skel;

  auto         vn  = normalize(v);
  auto         J   = diff(v);

  // critical points
  std::vector<mat<Real, 2, 2>> saddle_eigvecs;
  std::vector<vec<Real, 2>>    saddle_eigvals;
  for (const auto& cp : find_critical_points(v.sampler())) {
    auto [eigvecs, eigvals] = eigenvectors(J(cp));
    if (std::abs(eigvals(0).imag()) < 1e-10 &&
        std::abs(eigvals(1).imag()) < 1e-10) {
      // non-swirling behavior
      if (eigvals(0).real() < 0 && eigvals(1).real() < 0) {
        skel.sinks.push_back(cp);
      } else if (eigvals(0).real() > 0 && eigvals(1).real() > 0) {
        skel.sources.push_back(cp);
      } else {
        skel.saddles.push_back(cp);
        saddle_eigvecs.push_back(real(eigvecs));
        saddle_eigvals.push_back(real(eigvals));
      }
    } else {
      // swirling behavior
      if (eigvals(0).real() < -1e-10 && eigvals(1).real() < -1e-10) {
        skel.attracting_foci.push_back(cp);
      } else if (eigvals(0).real() > 1e-10 && eigvals(1).real() > 1e-10) {
        skel.repelling_foci.push_back(cp);
      } else {
        skel.centers.push_back(cp);
      }
    }
  }
  // create separatrices starting from saddle points
  auto saddle_eigvecs_it = begin(saddle_eigvecs);
  auto saddle_eigvals_it = begin(saddle_eigvals);
  for (const auto& saddle : skel.saddles) {
    const auto& eigvecs           = *(saddle_eigvecs_it++);
    const auto& eigvals           = *(saddle_eigvals_it++);
    for (size_t i = 0; i < 2; ++i) {
      const double cur_tau = eigvals(i) > 0 ? tau : -tau;
      skel.separatrices.push_back(integrator.integrate(
          vn, saddle + normalize(eigvecs.col(i)) * eps, 0, cur_tau));
      skel.separatrices.push_back(integrator.integrate(
          vn, saddle - normalize(eigvecs.col(i)) * eps, 0, cur_tau));
    }
  }

  // boundary switch points
  for (const auto& bsp : find_boundary_switch_points(v)) {
    skel.boundary_switch_points.push_back(bsp);

    // calculate separatrices if flowing into domain
    auto a         = J(bsp) * v(bsp);
    bool integrate = true;

    if (bsp(0) == v.sampler().dimension(0).front()) {
      if (a(0) < 0) { integrate = false; }
    } else if (bsp(0) == v.sampler().dimension(0).back()) {
      if (a(0) > 0) { integrate = false; }
    } else if (bsp(1) == v.sampler().dimension(1).front()) {
      if (a(1) < 0) { integrate = false; }
    } else if (bsp(1) == v.sampler().dimension(1).back()) {
      if (a(1) > 0) { integrate = false; }
    }

    if (integrate) {
      skel.separatrices.push_back(integrator.integrate(vn, bsp, 0, tau));
      skel.separatrices.push_back(integrator.integrate(vn, bsp, 0, -tau));
    }
  }
  return skel;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
