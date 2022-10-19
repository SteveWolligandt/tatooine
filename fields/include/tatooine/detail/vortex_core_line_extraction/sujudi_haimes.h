#ifndef TATOOINE_FIELDS_DETAIL_VORTEX_CORE_LINE_EXTRACTION_SUJUDI_HAIMES_H
#define TATOOINE_FIELDS_DETAIL_VORTEX_CORE_LINE_EXTRACTION_SUJUDI_HAIMES_H
//==============================================================================
#include <tatooine/differentiated_field.h>
#include <tatooine/field_operations.h>
#include <tatooine/parallel_vectors.h>
//==============================================================================
namespace tatooine::detail::vortex_core_lines {
//==============================================================================
template <typename Field, typename Real, typename DomainX, typename DomainY,
          typename DomainZ>
auto sujudi_haimes(
    vectorfield<Field, Real, 3> const& v, integral auto const t,
    tatooine::rectilinear_grid<DomainX, DomainY, DomainZ> const& g) {
  auto       J                            = diff(v);
  auto const filter_imaginary_eigenvalues = [&](auto const& x) {
    auto       jxt     = J(x, t);
    auto const lambdas = eigenvalues(jxt);
    for (auto const lambda : lambdas) {
      if (abs(lambda.imag()) > 1e-10) {
        return true;
      }
    }
    return false;
  };
  return parallel_vectors(v, J * v, g, t, execution_policy::parallel,
                          filter_imaginary_eigenvalues);
}
//==============================================================================
template <typename Grid, typename Real, bool HasNonConstReference>
auto sujudi_haimes(rectilinear_grid::typed_vertex_property_interface<
                   Grid, vec<Real, 3>, HasNonConstReference> const& v) {
  auto J = diff(v);
  // auto const filter_imaginary_eigenvalues = [&](auto const& x) {
  //   auto       jxt     = J(x, t);
  //   auto const lambdas = eigenvalues(jxt);
  //   for (auto const lambda : lambdas) {
  //     if (abs(lambda.imag()) > 1e-10) {
  //       return true;
  //     }
  //   }
  //   return false;
  // };
  return detail::calc_parallel_vectors<Real>(
      // get vf data by evaluating V field
      [&](auto const ix, auto const iy, auto const iz, auto const& /*p*/) {
        return v(ix, iy, iz);
      },
      // get wf data by evaluating W field
      [&](auto const ix, auto const iy, auto const iz, auto const& /*p*/) {
        return J(ix, iy, iz) * v(ix, iy, iz);
      },
      v.grid(), execution_policy::parallel);
}
//==============================================================================
template <typename Grid, typename Real, bool HasNonConstReference>
auto sujudi_haimes(rectilinear_grid::typed_vertex_property_interface<
                       Grid, Vec3<Real>, HasNonConstReference> const& v,
                   rectilinear_grid::typed_vertex_property_interface<
                       Grid, Mat3<Real>, HasNonConstReference> const& J) {
  auto       J_sampler                    = J.linear_sampler();
  auto const filter_imaginary_eigenvalues = [&](auto const& x) {
    auto       jxt     = J_sampler(x);
    auto const lambdas = eigenvalues(jxt);
    if (jxt.isnan()) {
      std::cout << "==========\n";
      std::cout << x << '\n';
      std::cout << jxt << '\n';
      std::cout << lambdas << '\n';
    }
    for (auto const lambda : lambdas) {
      if (abs(lambda.imag()) > 1e-10) {
        return true;
      }
    }
    return false;
  };
  return detail::calc_parallel_vectors<Real>(
      // get vf data by evaluating V field
      [&](auto const ix, auto const iy, auto const iz, auto const& /*p*/) {
        return v(ix, iy, iz);
      },
      // get wf data by evaluating W field
      [&](auto const ix, auto const iy, auto const iz, auto const& /*p*/) {
        return J(ix, iy, iz) * v(ix, iy, iz);
      },
      v.grid(), execution_policy::parallel, filter_imaginary_eigenvalues);
}
//==============================================================================
}  // namespace tatooine::detail::vortex_core_lines
//==============================================================================
#endif
