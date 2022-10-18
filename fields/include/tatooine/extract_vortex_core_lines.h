#ifndef TATOOINE_FIELDS_EXTRACT_VORTEX_CORE_LINES_H
#define TATOOINE_FIELDS_EXTRACT_VORTEX_CORE_LINES_H
//==============================================================================
#include <tatooine/differentiated_field.h>
#include <tatooine/field_operations.h>
#include <tatooine/parallel_vectors.h>
//==============================================================================
namespace tatooine {
//==============================================================================
namespace detail::vortex_core_lines {
//==============================================================================
template <typename Field, typename Real, typename DomainX, typename DomainY,
          typename DomainZ>
auto sujudi_haimes(
    vectorfield<Field, Real, 3> const& v, integral auto const t,
    tatooine::rectilinear_grid<DomainX, DomainY, DomainZ> const& g) {
  auto       J                     = diff(v);
  auto const imaginary_eigenvalues = [&](auto const& x) {
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
                          imaginary_eigenvalues);
}
//==============================================================================
template <typename Grid, typename Real, bool HasNonConstReference>
auto sujudi_haimes(rectilinear_grid::typed_vertex_property_interface<
                   Grid, vec<Real, 3>, HasNonConstReference> const& v) {
  auto J = diff(v);
  // auto const imaginary_eigenvalues = [&](auto const& x) {
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
}  // namespace detail::vortex_core_lines
//==============================================================================
namespace algorithm {
//==============================================================================
struct sujudi_haimes_tag {};
static constexpr auto sujudi_haimes = sujudi_haimes_tag{};
//==============================================================================
}  // namespace algorithm
//==============================================================================
template <typename Field, typename Real, typename DomainX, typename DomainY,
          typename DomainZ>
auto extract_vortex_core_lines(
    vectorfield<Field, Real, 3> const& v, integral auto const t,
    rectilinear_grid<DomainX, DomainY, DomainZ> const& g,
    algorithm::sujudi_haimes_tag const /*algorithm*/) {
  return detail::vortex_core_lines::sujudi_haimes(v, t, g);
}
//------------------------------------------------------------------------------
template <typename Field, typename Real, typename DomainX, typename DomainY,
          typename DomainZ>
auto extract_vortex_core_lines(
    vectorfield<Field, Real, 3> const& v, integral auto const t,
    rectilinear_grid<DomainX, DomainY, DomainZ> const& g) {
  return extract_vortex_core_lines(v, t, algorithm::sujudi_haimes);
}
//==============================================================================
template <typename Grid, typename Real, bool HasNonConstReference>
auto extract_vortex_core_lines(
    detail::rectilinear_grid::typed_vertex_property_interface<
        Grid, vec<Real, 3>, HasNonConstReference> const& v,
    algorithm::sujudi_haimes_tag const /*algorithm*/) {
  return detail::vortex_core_lines::sujudi_haimes(v);
}
//------------------------------------------------------------------------------
template <typename Grid, typename Real, bool HasNonConstReference>
auto extract_vortex_core_lines(
    detail::rectilinear_grid::typed_vertex_property_interface<
        Grid, vec<Real, 3>, HasNonConstReference> const& v) {
  return extract_vortex_core_lines(v, algorithm::sujudi_haimes);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
