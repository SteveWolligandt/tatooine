#ifndef TATOOINE_FIELDS_VORTEX_CORE_LINES_EXTRACTION_H
#define TATOOINE_FIELDS_VORTEX_CORE_LINES_EXTRACTION_H
//==============================================================================
#include <tatooine/detail/vortex_core_line_extraction/sujudi_haimes.h>
//==============================================================================
namespace tatooine::algorithm {
//==============================================================================
struct sujudi_haimes_tag {};
static constexpr auto sujudi_haimes = sujudi_haimes_tag{};
//==============================================================================
}  // namespace tatooine::algorithm
//==============================================================================
namespace tatooine {
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
        Grid, Vec3<Real>, HasNonConstReference> const& v,
        detail::rectilinear_grid::typed_vertex_property_interface<
        Grid, Mat3<Real>, HasNonConstReference> const& J,
    algorithm::sujudi_haimes_tag const /*algorithm*/) {
  return detail::vortex_core_lines::sujudi_haimes(v, J);
}
//------------------------------------------------------------------------------
template <typename Grid, typename Real, bool HasNonConstReference>
auto extract_vortex_core_lines(
    detail::rectilinear_grid::typed_vertex_property_interface<
        Grid, vec<Real, 3>, HasNonConstReference> const& v) {
  return extract_vortex_core_lines(v, algorithm::sujudi_haimes);
}
//------------------------------------------------------------------------------
template <typename Grid, typename Real, bool HasNonConstReference>
auto extract_vortex_core_lines(
    detail::rectilinear_grid::typed_vertex_property_interface<
        Grid, Vec3<Real>, HasNonConstReference> const& v,
        detail::rectilinear_grid::typed_vertex_property_interface<
        Grid, Mat3<Real>, HasNonConstReference> const& J) {
  return extract_vortex_core_lines(v, J, algorithm::sujudi_haimes);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
