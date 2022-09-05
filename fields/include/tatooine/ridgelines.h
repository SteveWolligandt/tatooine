#ifndef TATOOINE_FIELDS_RIDGELINES_H
#define TATOOINE_FIELDS_RIDGELINES_H
//==============================================================================
#include <tatooine/rectilinear_grid.h>
#include <tatooine/edgeset.h>
//==============================================================================
namespace tatooine {
//==============================================================================
//template <typename XDomain, typename YDomain>
//auto ridgelines(
//    invocable<
//        std::size_t, std::size_t,
//        Vec2<typename rectilinear_grid<XDomain, YDomain>::real_type>> auto&&
//                                              get_scalars,
//    rectilinear_grid<XDomain, YDomain> const& g) {
//  using real_type    = typename rectilinear_grid<XDomain, YDomain>::real_type;
//  using edgeset_type = Edgeset2<real_type>;
//  auto ridgelines    = edgeset_type{};
//
//#ifdef NDEBUG
//  auto mutex = std::mutex{};
//#endif
//  auto const execution =
//#ifdef NDEBUG
//      execution_policy::sequential;
//#else
//      execution_policy::parallel;
//#endif
//  auto iteration = [&](std::size_t const ix, std::size_t const iy) {
//    
//  };
//  for_loop(iteration, execution, g.size(0) - 1, g.size(1) - 1);
//  return ridgelines;
//}
////------------------------------------------------------------------------------
template <typename Grid, arithmetic T, bool HasNonConstReference>
auto ridgelines(detail::rectilinear_grid::typed_vertex_property_interface<
                Grid, T, HasNonConstReference> const& data) {
  using real_type    = typename Grid::real_type;
  using edgeset_type = Edgeset2<real_type>;
  auto ridgelines    = edgeset_type{};
//  diff<2>(data);
//
//#ifdef NDEBUG
//  auto mutex = std::mutex{};
//#endif
//  auto const execution =
//#ifdef NDEBUG
//      execution_policy::sequential;
//#else
//      execution_policy::parallel;
//#endif
//  auto iteration = [&](std::size_t const ix, std::size_t const iy) {
//
//  };
//  for_loop(iteration, execution, g.size(0) - 1, g.size(1) - 1);
  return ridgelines;
}
////------------------------------------------------------------------------------
//template <arithmetic Real, typename Indexing, arithmetic BBReal>
//auto ridgelines(dynamic_multidim_array<Real, Indexing> const& data,
//              axis_aligned_bounding_box<BBReal, 2> const&   bb) {
//  assert(data.num_dimensions() == 2);
//  return ridgelines(
//      [&](auto ix, auto iy, auto const& /*ps*/) -> auto const& {
//        return data(ix, iy);
//      },
//      rectilinear_grid{linspace{bb.min(0), bb.max(0), data.size(0)},
//                       linspace{bb.min(1), bb.max(1), data.size(1)}}
//      );
//}
////------------------------------------------------------------------------------
//template <arithmetic Real, arithmetic  BBReal, typename Indexing,
//          typename MemLoc, std::size_t XRes, std::size_t YRes>
//auto ridgelines(
//    static_multidim_array<Real, Indexing, MemLoc, XRes, YRes> const& data,
//    axis_aligned_bounding_box<BBReal, 2> const&                      bb) {
//  return ridgelines(
//      [&](auto ix, auto iy, auto const& /*ps*/) -> auto const& {
//        return data(ix, iy);
//      },
//      rectilinear_grid{linspace{bb.min(0), bb.max(0), data.size(0)},
//                       linspace{bb.min(1), bb.max(1), data.size(1)}}
//      );
//}
//------------------------------------------------------------------------------
//template <typename Field, typename FieldReal,
//          detail::rectilinear_grid::dimension XDomain,
//          detail::rectilinear_grid::dimension YDomain,
//          arithmetic                          TReal = FieldReal>
//auto ridgelines(scalarfield<Field, FieldReal, 2> const&   sf,
//                rectilinear_grid<XDomain, YDomain> const& g,
//                TReal const                               t = 0) {
//  auto eval = [&](auto const /*ix*/, auto const /*iy*/, auto const& pos) {
//    return sf(pos, t);
//  };
//  return ridgelines(eval, g);
//}
//==============================================================================
}
//==============================================================================
#endif
